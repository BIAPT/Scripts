#!/usr/bin/env ruby
#the final version I am working on

role = "user"
#role = "caregiver"


for foldername in Dir["TP00*"]
  #puts foldername
  for filename in Dir[foldername + "/2019-*.csv"]
    if !filename.include? "setting"
      puts filename
      filename2 = role + "." + filename[/.*\_(.*?)\_/,1] + ".stream~"
      puts filename2


      #filename2=filename.sub(/csv$/,"stream~")
      raise "big problem" if filename==filename2
      f=File.open(filename,"r")
      g=File.open(foldername+"/"+filename2, "w")
      f.gets
      rown = 0
      while line=f.gets
        line1 = line.split(/;/)[1,2].join(';')
        line2 = line1.delete("\"").to_f
        g.puts line2
        rown = rown + 1
      end
      g.close
      
      
      tnow = Time.now.strftime ("%Y/%m/%d %H:%M:%S:%L").to_s
      tutc = Time.now.getgm.strftime ("%Y/%m/%d %H:%M:%S:%L").to_s
      filename3=filename2.sub(/stream~/,"stream")
      s=File.open(foldername+"/"+filename3,"w")

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
      s.puts "<?xml version=\"1.0\" ?>"
      s.puts "<stream ssi-v=\"2\">"
      s.puts tab + "<info ftype=\"ASCII\" sr=\"" + samplerate.to_s + "\" dim=\"1\" byte=\"4\" type=\"FLOAT\" delim=\";\" />"
      s.puts tab + "<meta />" # Not sure why we need this XML tag?
      s.puts tab + "<time ms=\"0\" local=\"" + tnow + "\" system=\"" + tutc + "\"/>"
      s.puts tab + "<chunk from=\"0.000000\" to=\"" + (rown/samplerate).to_s + "\" byte=\"0\" num=\"" + rown.to_s + "\"/>"
      s.puts "</stream>"

      # closing our file pointer
      s.close
      f.close

    end
  end
end





