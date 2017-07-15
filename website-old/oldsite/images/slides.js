var Slides = Class.create({

	initialize: function(element, options) {		
		this.options = {
      		Duration: 1,
			Delay: 10.0,
			Random: true,
			Slideshow:true,
			Controls:true
    	}
		Object.extend(this.options, options || {});

    	this.element        = $(element);
		this.slides			= this.element.childElements();
		this.num_slides		= this.slides.length;		
		this.current_slide 	= (this.options.Random) ? (Math.floor(Math.random()*this.num_slides)) : 0;
		this.end_slide		= this.num_slides - 1;
		
		this.slides.invoke('hide');
		this.slides[this.current_slide].show();
				
		if (this.options.Slideshow) { 
			this.startSlideshow();
		}				
		if (this.options.Controls) {
			this.addControls();
		}		
	},
	
	addControls: function() {
		this.btn_previous	= $('previous');
		this.btn_next 		= $('next');
		this.btn_start		= $('start');
		this.btn_stop		= $('stop');
		
		this.btn_previous.observe('click', this.moveToPrevious.bindAsEventListener(this));
		this.btn_next.observe('click', this.moveToNext.bindAsEventListener(this));
		this.btn_start.observe('click', this.startSlideshow.bindAsEventListener(this));
		this.btn_stop.observe('click', this.stopSlideshow.bindAsEventListener(this));
	},

	startSlideshow: function(event) {
		if (event) { Event.stop(event); }
		if (!this.running)	{
			this.fadeStartBtn();
			this.executer = new PeriodicalExecuter(function(){
	  			this.updateSlide(this.current_slide+1);
	 		}.bind(this),this.options.Delay);
			this.running=true;
		}
		
	},
	
	fadeStartBtn: function() {
		var startBtn = $('start');
		var stopBtn = $('stop');
		Effect.Fade(startBtn, { duration: 0.3 }),
		Effect.Appear(stopBtn, { duration: 0.3 }) 
	},
	
	stopSlideshow: function(event) {	
		if (event) { Event.stop(event); } 
		if (this.executer) {
			this.fadeStopBtn();
			this.executer.stop();
			this.running=false;
		}	 
	},
	
	fadeStopBtn: function() {
		var startBtn = $('start');
		var stopBtn = $('stop');
		Effect.Fade(stopBtn, { duration: 0.3 }),
		Effect.Appear(startBtn, { duration: 0.3 }) 
	},

	moveToPrevious: function (event) {
		if (event) { Event.stop(event); }
		//this.stopSlideshow();
  		this.updateSlide(this.current_slide-1);
	},

	moveToNext: function (event) {
		if (event) { Event.stop(event); }
		//this.stopSlideshow();
  		this.updateSlide(this.current_slide+1);
	},
	
	updateSlide: function(next_slide) {
		if (next_slide > this.end_slide) { 
				next_slide = 0; 
		} 
		else if ( next_slide == -1 ) {
				next_slide = this.end_slide;
		}
		
		this.fadeInOut(next_slide, this.current_slide);		
	},

 	fadeInOut: function (next, current) {		
		new Effect.Parallel([
			new Effect.Fade(this.slides[current], { sync: true }),
			new Effect.Appear(this.slides[next], { sync: true }) 
  		], { duration: this.options.Duration });
		
		this.current_slide = next;		
	}

});