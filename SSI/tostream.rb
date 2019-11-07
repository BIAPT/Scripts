#!/usr/bin/env ruby
# Author: Parisa
# Edit: Yacine
#the final version I am working on

# Get the timestamp of when the script was run 
# We do this here so that all file created when this is run 
# will have the same timestamp
tnow = Time.now.strftime("%Y/%m/%d %H:%M:%S:%L")
tutc = Time.now.getgm.strftime("%Y/%m/%d %H:%M:%S:%L")

# Iterating through each TPS folder
for tpsid in Dir["TP00*"]
  puts "TPS id: " + tpsid

  # Iterating through each files in the TPS folder
  for filename in Dir[tpsid + "/2019-*.csv"]

    # skip the setting file
    if filename.include? "setting"
      next
    end

    # Creating the Stream~ file (contianing the data)
    puts "Input file: " + filename
    streamname = tpsid + "." + filename[/.*\_(.*?)\_/,1] + ".stream~"
    puts "Output file: " + streamname

    # Some error checking
    raise "big problem" if filename == streamname

    # open our input and output file
    rawfile = File.open(filename, "r")
    streamfile = File.open(tpsid + "/" + streamname, "w")
    
    # Read the input file line by line 
    # and copy the data in the stream~ file in the right format
    rawfile.gets # skip the header of the input
    rown = 0 # keep track of the number of row written for the stream file
    while line = rawfile.gets
      # Not sure why we need these two manipulation (the first one seems fine)
      line = line.split(/;/)[1,2].join(';')
      datapoint = line.delete("\"").to_f

      streamfile.puts datapoint # write the number to the stream~ file
      rown = rown + 1
    end

    # Close our file pointers for the input data and the stream~
    streamfile.close
    rawfile.close
  
    # Creating the header stream file (no tilda ~)
    headername=streamname.sub(/stream~/,"stream")
    headerfile = File.open(tpsid + "/" + headername,"w")

    # Selecting right sample rate given the signal
    if filename.include? "BVP"
      samplerate = 300.0
    elsif filename.include? "EDA" or filename.include? "TEMP" or filename.include? "HR"
      samplerate = 15.0
    elsif filename.include? "STR"
      samplerate = -1.0 # What do we do with this?
    else
      raise "Filename doesn't include BVP, EDA, TEMP, HR or STR! Please double check if this code is still valid for your raw signals."
    end

    # Writing to the stream file
    tab = "\t" # for readability
    headerfile.puts "<?xml version=\"1.0\" ?>"
    headerfile.puts "<stream ssi-v=\"2\">"
    headerfile.puts tab + "<info ftype=\"ASCII\" sr=\"" + samplerate.to_s + "\" dim=\"1\" byte=\"4\" type=\"FLOAT\" delim=\";\" />"
    headerfile.puts tab + "<meta />" # Not sure why we need this XML tag?
    headerfile.puts tab + "<time ms=\"0\" local=\"" + tnow + "\" system=\"" + tutc + "\"/>"
    headerfile.puts tab + "<chunk from=\"0.000000\" to=\"" + (rown/samplerate).to_s + "\" byte=\"0\" num=\"" + rown.to_s + "\"/>"
    headerfile.puts "</stream>"

    # closing our file pointer
    headerfile.close
  end
end





