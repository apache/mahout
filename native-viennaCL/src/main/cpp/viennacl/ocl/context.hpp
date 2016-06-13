#ifndef VIENNACL_OCL_CONTEXT_HPP_
#define VIENNACL_OCL_CONTEXT_HPP_

/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/ocl/context.hpp
    @brief Represents an OpenCL context within ViennaCL
*/

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <fstream>
#include <vector>
#include <map>
#include <cstdlib>
#include "viennacl/ocl/forwards.h"
#include "viennacl/ocl/error.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/program.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/command_queue.hpp"
#include "viennacl/tools/sha1.hpp"
#include "viennacl/tools/shared_ptr.hpp"
namespace viennacl
{
namespace ocl
{
/** @brief Manages an OpenCL context and provides the respective convenience functions for creating buffers, etc.
  *
  * This class was originally written before the OpenCL C++ bindings were standardized.
  * Regardless, it provides a couple of convience functionality which is not covered by the OpenCL C++ bindings.
*/
class context
{
  typedef std::vector< tools::shared_ptr<viennacl::ocl::program> >   program_container_type;

public:
  context() : initialized_(false),
    device_type_(CL_DEVICE_TYPE_DEFAULT),
    current_device_id_(0),
    default_device_num_(1),
    pf_index_(0),
    current_queue_id_(0)
  {
    if (std::getenv("VIENNACL_CACHE_PATH"))
      cache_path_ = std::getenv("VIENNACL_CACHE_PATH");
    else
      cache_path_ = "";
  }

  //////// Get and set kernel cache path */
  /** @brief Returns the compiled kernel cache path */
  std::string cache_path() const { return cache_path_; }

  /** @brief Sets the compiled kernel cache path */
  void cache_path(std::string new_path) { cache_path_ = new_path; }

  //////// Get and set default number of devices per context */
  /** @brief Returns the maximum number of devices to be set up for the context */
  vcl_size_t default_device_num() const { return default_device_num_; }

  /** @brief Sets the maximum number of devices to be set up for the context */
  void default_device_num(vcl_size_t new_num) { default_device_num_ = new_num; }

  ////////// get and set preferred device type /////////////////////
  /** @brief Returns the default device type for the context */
  cl_device_type default_device_type()
  {
    return device_type_;
  }

  /** @brief Sets the device type for this context */
  void default_device_type(cl_device_type dtype)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Setting new device type for context " << h_ << std::endl;
#endif
    if (!initialized_)
      device_type_ = dtype; //assume that the user provided a correct value
  }

  //////////////////// get devices //////////////////
  /** @brief Returns a vector with all devices in this context */
  std::vector<viennacl::ocl::device> const & devices() const
  {
    return devices_;
  }

  /** @brief Returns the current device */
  viennacl::ocl::device const & current_device() const
  {
    //std::cout << "Current device id in context: " << current_device_id_ << std::endl;
    return devices_[current_device_id_];
  }

  /** @brief Switches the current device to the i-th device in this context */
  void switch_device(vcl_size_t i)
  {
    assert(i < devices_.size() && bool("Provided device index out of range!"));
    current_device_id_ = i;
  }

  /** @brief If the supplied device is used within the context, it becomes the current active device. */
  void switch_device(viennacl::ocl::device const & d)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Setting new current device for context " << h_ << std::endl;
#endif
    bool found = false;
    for (vcl_size_t i=0; i<devices_.size(); ++i)
    {
      if (devices_[i] == d)
      {
        found = true;
        current_device_id_ = i;
        break;
      }
    }
    if (found == false)
      std::cerr << "ViennaCL: Warning: Could not set device " << d.name() << " for context." << std::endl;
  }

  /** @brief Add a device to the context. Must be done before the context is initialized */
  void add_device(viennacl::ocl::device const & d)
  {
    assert(!initialized_ && bool("Device must be added to context before it is initialized!"));
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Adding new device to context " << h_ << std::endl;
#endif
    if (std::find(devices_.begin(), devices_.end(), d) == devices_.end())
      devices_.push_back(d);
  }

  /** @brief Add a device to the context. Must be done before the context is initialized */
  void add_device(cl_device_id d)
  {
    assert(!initialized_ && bool("Device must be added to context before it is initialized!"));
    add_device(viennacl::ocl::device(d));
  }


  /////////////////////// initialize context ///////////////////

  /** @brief Initializes a new context */
  void init()
  {
    init_new();
  }

  /** @brief Initializes the context from an existing, user-supplied context */
  void init(cl_context c)
  {
    init_existing(c);
  }

  /*        void existing_context(cl_context context_id)
    {
      assert(!initialized_ && bool("ViennaCL: FATAL error: Provided a new context for an already initialized context."));
      #i#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
      std::cout << "ViennaCL: Reusing existing context " << h_ << std::endl;
      #e#endif
      h_ = context_id;
    }*/

  ////////////////////// create memory /////////////////////////////

  /** @brief Creates a memory buffer within the context. Does not wrap the OpenCL handle into the smart-pointer-like viennacl::ocl::handle, which saves an OpenCL backend call, yet the user has to ensure that the OpenCL memory handle is free'd or passed to a viennacl::ocl::handle later on.
    *
    *  @param flags  OpenCL flags for the buffer creation
    *  @param size   Size of the memory buffer in bytes
    *  @param ptr    Optional pointer to CPU memory, with which the OpenCL memory should be initialized
    *  @return       A plain OpenCL handle. Either assign it to a viennacl::ocl::handle<cl_mem> directly, or make sure that you free to memory manually if you no longer need the allocated memory.
    */
  cl_mem create_memory_without_smart_handle(cl_mem_flags flags, unsigned int size, void * ptr = NULL) const
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Creating memory of size " << size << " for context " << h_ << " (unsafe, returning cl_mem directly)" << std::endl;
#endif
    if (ptr)
      flags |= CL_MEM_COPY_HOST_PTR;
    cl_int err;
    cl_mem mem = clCreateBuffer(h_.get(), flags, size, ptr, &err);
    VIENNACL_ERR_CHECK(err);
    return mem;
  }


  /** @brief Creates a memory buffer within the context
    *
    *  @param flags  OpenCL flags for the buffer creation
    *  @param size   Size of the memory buffer in bytes
    *  @param ptr    Optional pointer to CPU memory, with which the OpenCL memory should be initialized
    */
  viennacl::ocl::handle<cl_mem> create_memory(cl_mem_flags flags, unsigned int size, void * ptr = NULL) const
  {
    return viennacl::ocl::handle<cl_mem>(create_memory_without_smart_handle(flags, size, ptr), *this);
  }

  /** @brief Creates a memory buffer within the context initialized from the supplied data
    *
    *  @param flags  OpenCL flags for the buffer creation
    *  @param buffer A vector (STL vector, ublas vector, etc.)
    */
  template< typename NumericT, typename A, template<typename, typename> class VectorType >
  viennacl::ocl::handle<cl_mem> create_memory(cl_mem_flags flags, const VectorType<NumericT, A> & buffer) const
  {
    return viennacl::ocl::handle<cl_mem>(create_memory_without_smart_handle(flags, static_cast<cl_uint>(sizeof(NumericT) * buffer.size()), (void*)&buffer[0]), *this);
  }

  //////////////////// create queues ////////////////////////////////

  /** @brief Adds an existing queue for the given device to the context */
  void add_queue(cl_device_id dev, cl_command_queue q)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Adding existing queue " << q << " for device " << dev << " to context " << h_ << std::endl;
#endif
    viennacl::ocl::handle<cl_command_queue> queue_handle(q, *this);
    queues_[dev].push_back(viennacl::ocl::command_queue(queue_handle));
    queues_[dev].back().handle().inc();
  }

  /** @brief Adds a queue for the given device to the context */
  void add_queue(cl_device_id dev)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Adding new queue for device " << dev << " to context " << h_ << std::endl;
#endif
      cl_int err;
#ifdef VIENNACL_PROFILING_ENABLED
    viennacl::ocl::handle<cl_command_queue> temp(clCreateCommandQueue(h_.get(), dev, CL_QUEUE_PROFILING_ENABLE, &err), *this);
#else
    viennacl::ocl::handle<cl_command_queue> temp(clCreateCommandQueue(h_.get(), dev, 0, &err), *this);
#endif
    VIENNACL_ERR_CHECK(err);

    queues_[dev].push_back(viennacl::ocl::command_queue(temp));
  }

  /** @brief Adds a queue for the given device to the context */
  void add_queue(viennacl::ocl::device d) { add_queue(d.id()); }

  //get queue for default device:
  viennacl::ocl::command_queue & get_queue()
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Getting queue for device " << devices_[current_device_id_].name() << " in context " << h_ << std::endl;
    std::cout << "ViennaCL: Current queue id " << current_queue_id_ << std::endl;
#endif

    return queues_[devices_[current_device_id_].id()][current_queue_id_];
  }

  viennacl::ocl::command_queue const & get_queue() const
  {
    typedef std::map< cl_device_id, std::vector<viennacl::ocl::command_queue> >    QueueContainer;

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Getting const queue for device " << devices_[current_device_id_].name() << " in context " << h_ << std::endl;
    std::cout << "ViennaCL: Current queue id " << current_queue_id_ << std::endl;
#endif

    // find queue:
    QueueContainer::const_iterator it = queues_.find(devices_[current_device_id_].id());
    if (it != queues_.end()) {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
      std::cout << "ViennaCL: Queue handle " << (it->second)[current_queue_id_].handle() << std::endl;
#endif
      return (it->second)[current_queue_id_];
    }

    throw queue_not_found("Could not obtain current command queue");

    //return ((*it)->second)[current_queue_id_];
  }

  //get a particular queue:
  /** @brief Returns the queue with the provided index for the given device */
  viennacl::ocl::command_queue & get_queue(cl_device_id dev, vcl_size_t i = 0)
  {
    if (i >= queues_[dev].size())
      throw invalid_command_queue();

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Getting queue " << i << " for device " << dev << " in context " << h_ << std::endl;
#endif
    unsigned int device_index;
    for (device_index = 0; device_index < devices_.size(); ++device_index)
    {
      if (devices_[device_index] == dev)
        break;
    }

    assert(device_index < devices_.size() && bool("Device not within context"));

    return queues_[devices_[device_index].id()][i];
  }

  /** @brief Returns the current device */
  // TODO: work out the const issues
  viennacl::ocl::command_queue const & current_queue() //const
  {
    return queues_[devices_[current_device_id_].id()][current_queue_id_];
  }

  /** @brief Switches the current device to the i-th device in this context */
  void switch_queue(vcl_size_t i)
  {
    assert(i < queues_[devices_[current_device_id_].id()].size() && bool("In class 'context': Provided queue index out of range for device!"));
    current_queue_id_ = i;
  }

  /** @brief If the supplied command_queue is used within the context, it becomes the current active command_queue, the command_queue's device becomes current active device. */
  void switch_queue(viennacl::ocl::command_queue const & q)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Setting new current queue for context " << h_ << std::endl;
#endif
    bool found = false;
    typedef std::map< cl_device_id, std::vector<viennacl::ocl::command_queue> >    QueueContainer;

    // For each device:
    vcl_size_t j = 0;
    for (QueueContainer::const_iterator it=queues_.begin(); it != queues_.end(); it++,j++)
    {
      const std::vector<viennacl::ocl::command_queue> & qv = (it->second);
      // For each queue candidate
      for (vcl_size_t i=0; i<qv.size(); ++i)
      {
        if (qv[i] == q)
        {
          found = true;
          current_device_id_ = j;
          current_queue_id_ = i;
          break;
        }
      }
    }
    if (found == false)
      std::cerr << "ViennaCL: Warning: Could not set queue " << q.handle().get() << " for context." << std::endl;
  }

  /////////////////// create program ///////////////////////////////
  /** @brief Adds a program to the context
    */
  viennacl::ocl::program & add_program(cl_program p, std::string const & prog_name)
  {
    programs_.push_back(tools::shared_ptr<ocl::program>(new viennacl::ocl::program(p, *this, prog_name)));
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Adding program '" << prog_name << "' with cl_program to context " << h_ << std::endl;
#endif
    return *programs_.back();
  }

  /** @brief Adds a new program with the provided source to the context. Compiles the program and extracts all kernels from it
    */
  viennacl::ocl::program & add_program(std::string const & source, std::string const & prog_name)
  {
    const char * source_text = source.c_str();
    vcl_size_t source_size = source.size();
    cl_int err;

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Adding program '" << prog_name << "' with source to context " << h_ << std::endl;
#endif

    cl_program temp = 0;

    //
    // Retrieves the program in the cache
    //
    if (cache_path_.size())
    {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
      std::cout << "ViennaCL: Cache at " << cache_path_ << std::endl;
#endif

      std::string prefix;
      for(std::vector< viennacl::ocl::device >::const_iterator it = devices_.begin(); it != devices_.end(); ++it)
        prefix += it->name() + it->vendor() + it->driver_version();
      std::string sha1 = tools::sha1(prefix + source);

      std::ifstream cached((cache_path_+sha1).c_str(),std::ios::binary);
      if (cached)
      {
        vcl_size_t len;
        std::vector<unsigned char> buffer;

        cached.read((char*)&len, sizeof(vcl_size_t));
        buffer.resize(len);
        cached.read((char*)(&buffer[0]), std::streamsize(len));

        cl_int status;
        cl_device_id devid = devices_[0].id();
        const unsigned char * bufdata = &buffer[0];
        temp = clCreateProgramWithBinary(h_.get(),1,&devid,&len, &bufdata,&status,&err);
        VIENNACL_ERR_CHECK(err);
      }
    }

    if (!temp)
    {
      temp = clCreateProgramWithSource(h_.get(), 1, (const char **)&source_text, &source_size, &err);
      VIENNACL_ERR_CHECK(err);
    }

    const char * options = build_options_.c_str();
    err = clBuildProgram(temp, 0, NULL, options, NULL, NULL);
#ifndef VIENNACL_BUILD_INFO
    if (err != CL_SUCCESS)
#endif
    {
      cl_build_status status;
      clGetProgramBuildInfo(temp, devices_[0].id(), CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL);
      std::cout << "Build Status = " << status << " ( Err = " << err << " )" << std::endl;

      char *build_log;
      size_t ret_val_size; // don't use vcl_size_t here
      err = clGetProgramBuildInfo(temp, devices_[0].id(), CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
      build_log = new char[ret_val_size+1];
      err = clGetProgramBuildInfo(temp, devices_[0].id(), CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
      build_log[ret_val_size] = '\0';
      std::cout << "Log: " << build_log << std::endl;
      delete[] build_log;

      std::cout << "Sources: " << source << std::endl;
    }
    VIENNACL_ERR_CHECK(err);

    //
    // Store the program in the cache
    //
    if (cache_path_.size())
    {
      vcl_size_t len;

      std::vector<vcl_size_t> sizes(devices_.size());
      clGetProgramInfo(temp,CL_PROGRAM_BINARY_SIZES,0,NULL,&len);
      clGetProgramInfo(temp,CL_PROGRAM_BINARY_SIZES,len,(void*)&sizes[0],NULL);

      std::vector<unsigned char*> binaries;
      for (vcl_size_t i = 0; i < devices_.size(); ++i)
        binaries.push_back(new unsigned char[sizes[i]]);

      clGetProgramInfo(temp,CL_PROGRAM_BINARIES,0,NULL,&len);
      clGetProgramInfo(temp,CL_PROGRAM_BINARIES,len,&binaries[0],NULL);

      std::string prefix;
      for(std::vector< viennacl::ocl::device >::const_iterator it = devices_.begin(); it != devices_.end(); ++it)
        prefix += it->name() + it->vendor() + it->driver_version();
      std::string sha1 = tools::sha1(prefix + source);
      std::ofstream cached((cache_path_+sha1).c_str(),std::ios::binary);

      cached.write((char*)&sizes[0], sizeof(vcl_size_t));
      cached.write((char*)binaries[0], std::streamsize(sizes[0]));

      for (vcl_size_t i = 0; i < devices_.size(); ++i)
        delete[] binaries[i];

      VIENNACL_ERR_CHECK(err);
    }


    programs_.push_back(tools::shared_ptr<ocl::program>(new ocl::program(temp, *this, prog_name)));

    viennacl::ocl::program & prog = *programs_.back();

    //
    // Extract kernels
    //
    cl_kernel kernels[1024];
    cl_uint   num_kernels_in_prog;
    err = clCreateKernelsInProgram(prog.handle().get(), 1024, kernels, &num_kernels_in_prog);
    VIENNACL_ERR_CHECK(err);

    for (cl_uint i=0; i<num_kernels_in_prog; ++i)
    {
      char kernel_name[128];
      err = clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, 128, kernel_name, NULL);
      prog.add_kernel(kernels[i], std::string(kernel_name));
    }

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Stored program '" << programs_.back()->name() << "' in context " << h_ << std::endl;
    std::cout << "ViennaCL: There is/are " << programs_.size() << " program(s)" << std::endl;
#endif

    return prog;
  }

  /** @brief Delete the program with the provided name */
  void delete_program(std::string const & name)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Deleting program '" << name << "' from context " << h_ << std::endl;
#endif
    for (program_container_type::iterator it = programs_.begin();
         it != programs_.end();
         ++it)
    {
      if ((*it)->name() == name)
      {
        programs_.erase(it);
        return;
      }
    }
  }

  /** @brief Returns the program with the provided name */
  viennacl::ocl::program & get_program(std::string const & name)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Getting program '" << name << "' from context " << h_ << std::endl;
    std::cout << "ViennaCL: There are " << programs_.size() << " programs" << std::endl;
#endif
    for (program_container_type::iterator it = programs_.begin();
         it != programs_.end();
         ++it)
    {
      //std::cout << "Name: " << (*it)->name() << std::endl;
      if ((*it)->name() == name)
        return **it;
    }
    std::cerr << "ViennaCL: Could not find program '" << name << "'" << std::endl;
    throw program_not_found(name);
    //return programs_[0];  //return a defined object
  }

  viennacl::ocl::program const & get_program(std::string const & name) const
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Getting program '" << name << "' from context " << h_ << std::endl;
    std::cout << "ViennaCL: There are " << programs_.size() << " programs" << std::endl;
#endif
    for (program_container_type::const_iterator it = programs_.begin();
         it != programs_.end();
         ++it)
    {
      //std::cout << "Name: " << (*it)->name() << std::endl;
      if ((*it)->name() == name)
        return **it;
    }
    std::cerr << "ViennaCL: Could not find program '" << name << "'" << std::endl;
    throw program_not_found(name);
    //return programs_[0];  //return a defined object
  }

  /** @brief Returns whether the program with the provided name exists or not */
  bool has_program(std::string const & name)
  {
    for (program_container_type::iterator it = programs_.begin();
         it != programs_.end();
         ++it)
    {
      if ((*it)->name() == name) return true;
    }
    return false;
  }

  /** @brief Returns the program with the provided id */
  viennacl::ocl::program & get_program(vcl_size_t id)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Getting program '" << id << "' from context " << h_ << std::endl;
    std::cout << "ViennaCL: There are " << programs_.size() << " programs" << std::endl;
#endif

    if (id >= programs_.size())
      throw invalid_program();

    return *programs_[id];
  }

  program_container_type get_programs()
  {
    return programs_;
  }

  /** @brief Returns the number of programs within this context */
  vcl_size_t program_num() { return programs_.size(); }

  /** @brief Convenience function for retrieving the kernel of a program directly from the context */
  viennacl::ocl::kernel & get_kernel(std::string const & program_name, std::string const & kernel_name) { return get_program(program_name).get_kernel(kernel_name); }

  /** @brief Returns the number of devices within this context */
  vcl_size_t device_num() { return devices_.size(); }

  /** @brief Returns the context handle */
  const viennacl::ocl::handle<cl_context> & handle() const { return h_; }

  /** @brief Returns the current build option string */
  std::string build_options() const { return build_options_; }

  /** @brief Sets the build option string, which is passed to the OpenCL compiler in subsequent compilations. Does not effect programs already compiled previously. */
  void build_options(std::string op) { build_options_ = op; }

  /** @brief Returns the platform ID of the platform to be used for the context */
  vcl_size_t platform_index() const  { return pf_index_; }

  /** @brief Sets the platform ID of the platform to be used for the context */
  void platform_index(vcl_size_t new_index)
  {
    assert(!initialized_ && bool("Platform ID must be set before context is initialized!"));
    pf_index_ = new_index;
  }

  /** @brief Less-than comparable for compatibility with std:map  */
  bool operator<(context const & other) const
  {
    return h_.get() < other.h_.get();
  }

  bool operator==(context const & other) const
  {
    return h_.get() == other.h_.get();
  }

private:
  /** @brief Initialize a new context. Reuse any previously supplied information (devices, queues) */
  void init_new()
  {
    assert(!initialized_ && bool("ViennaCL FATAL error: Context already created!"));

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Initializing new ViennaCL context." << std::endl;
#endif

    cl_int err;
    std::vector<cl_device_id> device_id_array;
    if (devices_.empty()) //get the default device if user has not yet specified a list of devices
    {
      //create an OpenCL context for the provided devices:
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
      std::cout << "ViennaCL: Setting all devices for context..." << std::endl;
#endif

      platform pf(pf_index_);
      std::vector<device> devices = pf.devices(device_type_);
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
      std::cout << "ViennaCL: Number of devices for context: " << devices.size() << std::endl;
#endif
      vcl_size_t device_num = std::min<vcl_size_t>(default_device_num_, devices.size());
      for (vcl_size_t i=0; i<device_num; ++i)
        devices_.push_back(devices[i]);

      if (devices.size() == 0)
      {
        std::cerr << "ViennaCL: FATAL ERROR: No devices of type '";
        switch (device_type_)
        {
        case CL_DEVICE_TYPE_CPU:          std::cout << "CPU"; break;
        case CL_DEVICE_TYPE_GPU:          std::cout << "GPU"; break;
        case CL_DEVICE_TYPE_ACCELERATOR:  std::cout << "ACCELERATOR"; break;
        case CL_DEVICE_TYPE_DEFAULT:      std::cout << "DEFAULT"; break;
        default:
          std::cout << "UNKNOWN" << std::endl;
        }
        std::cout << "' found!" << std::endl;
      }
    }

    //extract list of device ids:
    for (std::vector< viennacl::ocl::device >::const_iterator iter = devices_.begin();
         iter != devices_.end();
         ++iter)
      device_id_array.push_back(iter->id());

    h_ = clCreateContext(0,
                         static_cast<cl_uint>(devices_.size()),
                         &(device_id_array[0]),
        NULL, NULL, &err);
    VIENNACL_ERR_CHECK(err);

    initialized_ = true;
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Initialization of new ViennaCL context done." << std::endl;
#endif
  }

  /** @brief Reuses a supplied context. */
  void init_existing(cl_context c)
  {
    assert(!initialized_ && bool("ViennaCL FATAL error: Context already created!"));
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Initialization of ViennaCL context from existing context." << std::endl;
#endif

    //set context handle:
    h_ = c;
    h_.inc(); // if the user provides the context, then the user will also call release() on the context. Without inc(), we would get a seg-fault due to double-free at program termination.

    if (devices_.empty())
    {
      //get devices for context:
      cl_int err;
      cl_uint num_devices;
      vcl_size_t temp;
      //Note: The obvious
      //  err = clGetContextInfo(h_, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);
      //does not work with NVIDIA OpenCL stack!
      err = clGetContextInfo(h_.get(), CL_CONTEXT_DEVICES, VIENNACL_OCL_MAX_DEVICE_NUM * sizeof(cl_device_id), NULL, &temp);
      VIENNACL_ERR_CHECK(err);
      assert(temp > 0 && bool("ViennaCL: FATAL error: Provided context does not contain any devices!"));
      num_devices = static_cast<cl_uint>(temp / sizeof(cl_device_id));

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
      std::cout << "ViennaCL: Reusing context with " << num_devices << " devices." << std::endl;
#endif

      std::vector<cl_device_id> device_ids(num_devices);
      err = clGetContextInfo(h_.get(), CL_CONTEXT_DEVICES, num_devices * sizeof(cl_device_id), &(device_ids[0]), NULL);
      VIENNACL_ERR_CHECK(err);

      for (vcl_size_t i=0; i<num_devices; ++i)
        devices_.push_back(viennacl::ocl::device(device_ids[i]));
    }
    current_device_id_ = 0;

    initialized_ = true;
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Initialization of ViennaCL context from existing context done." << std::endl;
#endif
  }


  bool initialized_;
  std::string cache_path_;
  cl_device_type device_type_;
  viennacl::ocl::handle<cl_context> h_;
  std::vector< viennacl::ocl::device > devices_;
  vcl_size_t current_device_id_;
  vcl_size_t default_device_num_;
  program_container_type programs_;
  std::map< cl_device_id, std::vector< viennacl::ocl::command_queue> > queues_;
  std::string build_options_;
  vcl_size_t pf_index_;
  vcl_size_t current_queue_id_;
}; //context



/** @brief Adds a kernel to the program */
inline viennacl::ocl::kernel & viennacl::ocl::program::add_kernel(cl_kernel kernel_handle, std::string const & kernel_name)
{
  assert(p_context_ != NULL && bool("Pointer to context invalid in viennacl::ocl::program object"));
  kernels_.push_back(tools::shared_ptr<ocl::kernel>(new ocl::kernel(kernel_handle, *this, *p_context_, kernel_name)));
  return *kernels_.back();
}

/** @brief Returns the kernel with the provided name */
inline viennacl::ocl::kernel & viennacl::ocl::program::get_kernel(std::string const & name)
{
  //std::cout << "Requiring kernel " << name << " from program " << name_ << std::endl;
  for (kernel_container_type::iterator it = kernels_.begin();
       it != kernels_.end();
       ++it)
  {
    if ((*it)->name() == name)
      return **it;
  }
  std::cerr << "ViennaCL: FATAL ERROR: Could not find kernel '" << name << "' from program '" << name_ << "'" << std::endl;
  std::cout << "Number of kernels in program: " << kernels_.size() << std::endl;
  throw kernel_not_found("Kernel not found");
  //return kernels_[0];  //return a defined object
}


inline void viennacl::ocl::kernel::set_work_size_defaults()
{
  assert( p_program_ != NULL && bool("Kernel not initialized, program pointer invalid."));
  assert( p_context_ != NULL && bool("Kernel not initialized, context pointer invalid."));

  if (   (p_context_->current_device().type() == CL_DEVICE_TYPE_GPU)
         || (p_context_->current_device().type() == CL_DEVICE_TYPE_ACCELERATOR) // Xeon Phi
         )
  {
    local_work_size_[0] = 128;      local_work_size_[1] = 0;  local_work_size_[2] = 0;
    global_work_size_[0] = 128*128; global_work_size_[1] = 0; global_work_size_[2] = 0;
  }
  else //assume CPU type:
  {
    //conservative assumption: one thread per CPU core:
    local_work_size_[0] = 1; local_work_size_[1] = 0; local_work_size_[2] = 0;

    size_type units = p_context_->current_device().max_compute_units();
    size_type s = 1;

    while (s < units) // find next power of 2. Important to make reductions work on e.g. six-core CPUs.
      s *= 2;

    global_work_size_[0] = s * local_work_size_[0]; global_work_size_[1] = 0; global_work_size_[2] = 0;
  }
}

}
}

#endif
