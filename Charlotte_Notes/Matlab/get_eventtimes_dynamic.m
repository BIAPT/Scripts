% Get the event-times of the N-400

sampling_rate= 250;
window_steps=0.1;
eventtimes=[];
eventlabel=[];

for i=1:length(EEG.event)
    % get the time of event in sec
    eventtimes(i)=EEG.event(i).latency/sampling_rate;
    if string(EEG.event(i).label)=="DIN85"
        eventlabel(i)=1; % incongruent
    elseif string(EEG.event(i).label)=="DIN95"
        eventlabel(i)=2; % congruent
    else
        eventlabel(i)=0; % no label
    end
    % transform and adapt to wpli analysis
    eventtimes(i)=round(eventtimes(i)*1/window_steps);
end

length(eventtimes)
length(eventlabel)

con=find(eventlabel==2);
incon=find(eventlabel==1);

length(incon)

inconeventtimes=eventtimes([incon])
coneventtimes=eventtimes([con])

save('N418_event_time_incon.mat','inconeventtimes')
save('N418_event_time_con.mat','coneventtimes')

