#!/usr/bin/perl -w

my $counter=0;
my $totalFiles=0;
my $step=1;
local $| = 1;

sub extract($) {
	my $file = shift;
	print "Now extracting: [$file]\n";
	system( "tar xf " . $file );
	my $ret = $?;
	if ( $ret == 0 ) {
		print "[$file] extracted successfully.\n";
	}
	else {
		print "Failed to extract file: [$file]\n";
	}
	return $ret;
}

sub convert(@) {
	my $id          = shift;
	my $outFileName = "User_history_$id.dat";

	local *outFile;
	local *inFile;
	open outFile, ">$outFileName"
	  or die "Cannot open [$outFileName] for writing.\n";
	for my $file (@_) {
		open inFile, "<training_set/$file" or die "Cannot open [training_set/$file] for reading.\n";
		my $movieId = <inFile>;
		my $userDetails;
		$movieId = substr $movieId, 0, length($movieId) - 2;
		while (<inFile>) {
			$userDetails = $_;
			$userDetails =~ s/,(\d),/"\t$movieId\t$1\t"/eg;
			print outFile $userDetails;
		}
		close inFile;
                $counter++;
                if( (($counter*100.0)/$totalFiles) >= $step) {
                	print "\nTotal Completed: $step %";
		      	$step++;
                }		
	}
	close outFile;
}

sub start($@) {
	my $parts    = shift;
	my @allFiles = @_;
	$totalFiles = @allFiles;
	my $partSize = ( $totalFiles / $parts );
        $partSize = $partSize < 1 ? 1 : $partSize;
	my $uid      = 0;
        print "Total files to be converted: [$totalFiles]\n";
	print "Staring data conversion ...";
	for ( my $start = 0 ; $start < $totalFiles ; $start += $partSize ) {
		convert( $uid, @allFiles[ $start .. ( $start + $partSize - 1 ) ] );
		$uid++;
	}

}

sub main {
	if ( extract("training_set.tar") == 0 ) {
		opendir DIR, "training_set";
		my @files = ( readdir DIR );
		for ( my $i = 0 ; $i < @files ; ) {
			if ( ( substr $files[$i], 0, 1 ) eq '.' ) {
				splice @files, $i, 1, ();
			}
			else {
				$i++;
			}
                }       
		start(10, @files );
		if($? ==0) {
    		  print ("\nCompleted!\n");
		  system("rm -rf training_set");
		} else {
		  print ("Data Conversion failed\n");
		}
	}
}

main
