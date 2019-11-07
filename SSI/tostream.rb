#!/usr/bin/env ruby
# Author: Parisa and Mathieu
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
  s.puts "<?xml version=\"1.0\" ?> \n <stream ssi-v=\"2\">"
  if filename.include? "BVP"
	samplerate = 300.0
  end

  if filename.include? "EDA"
	samplerate = 15.0
  end

  if filename.include? "TEMP"
	samplerate = 15.0
  end

  if filename.include? "HR"
	samplerate = 15.0
  end
s.puts "<info ftype=\"ASCII\" sr=\"" + samplerate.to_s + "\" dim=\"1\" byte=\"4\" type=\"FLOAT\" delim=\";\" />"


      
  s.puts "<meta />
  <time ms=\"0\" local=\"" + tnow + "\" system=\"" + tutc + "\"/>
  <chunk from=\"0.000000\" to=\"" + (rown/samplerate).to_s + "\" byte=\"0\" num=\"" + rown.to_s + "\"/>
</stream>"

s.close
f.close

end
end
end





