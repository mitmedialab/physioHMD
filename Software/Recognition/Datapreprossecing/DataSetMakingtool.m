TS=Cry3;
Mark=6;
Index1=8
TS=TS(:,1:end);
% TS(abs(TS)>1500)=0;
% TS1=TS(:,1:15000);
% TS1=[TS1,TS(:,25000:end)];
% % TS=[TS,Shock5];
% TS1=TS(:,1:46000);
% TS1=[TS1,TS(:,56000:end)];
% TS=TS1;
Temp=TS(1,:);
% Temp(abs(Temp)>200)=0;

% filter=fspecial('average',[1,30]); 
% % Temp=Temp-filter2(filter,Temp);
% Temp=filter2(filter,Temp);
% Temp(1,1:50)=0;
% Temp(1,end-50:end)=0;

S=size(TS);
Numb=S(1,2)/500;
Numb=round(Numb)-1;
TrainData=zeros(Numb,4,500);
TrainLabel=zeros(Numb,10);
LabelMark=zeros(1,S(1,2));

for i=1:4
%     TS(i,:)= mapminmax(TS(i,:),0.1,0.9);
end

%%  Mark

for i=1:Index1
plot(Temp)
h=imrect;  
pos= getPosition(h);
pos=round(pos);
LabelMark(1,pos(1):(pos(1)+pos(3)))=Mark;%% 
hold on
plot(pos(1):(pos(1)+pos(3)),zeros(1,pos(3)+1))
end

% Index2=2
% for i=1:Index2
% plot(Temp)
% h=imrect;  
% pos= getPosition(h);
% pos=round(pos);
% LabelMark(1,pos(1):(pos(1)+pos(3)))=6;%% 
% hold on
% plot(pos(1):(pos(1)+pos(3)),zeros(1,pos(3)+1))
% end

for i=1:Numb
    % segmentation
    TrainData(i,1,:)=TS(1,(500*(i-1)+1):500*i);
    TrainData(i,2,:)=TS(2,(500*(i-1)+1):500*i);
    TrainData(i,3,:)=TS(3,(500*(i-1)+1):500*i);
    TrainData(i,4,:)=TS(4,(500*(i-1)+1):500*i);   
    Marker(i,:)=LabelMark(1,(500*(i-1)+1):500*i); 
    % remove the mean
%     M=mean(TrainData(i,:,:),3);
%     TrainData(i,1,:)= TrainData(i,1,:)-M(1);
%     TrainData(i,2,:)= TrainData(i,2,:)-M(2);
%     TrainData(i,3,:)= TrainData(i,3,:)-M(3);
%     TrainData(i,4,:)= TrainData(i,4,:)-M(4);
end
%% Label

for i=1:Numb
SUM=sum(Marker(i,:));
    switch max(Marker(i,:)) 

        case 0
            TrainLabel(i,1)=1;  
        case 1
            if SUM >300
                TrainLabel(i,2)=1 ;   
            else
                TrainLabel(i,1)=1;
            end
            
        case 2
            if SUM>600
                TrainLabel(i,3)=1;
            else
                TrainLabel(i,1)=1;
            end            

        case 3
             if SUM >900
                 TrainLabel(i,4)=1 ; 
             else
                TrainLabel(i,1)=1;
             end
        case 4
             if SUM >1200
                 TrainLabel(i,5)=1 ; 
             else
                TrainLabel(i,1)=1;
             end
        case 5
             if SUM >1500
                 TrainLabel(i,6)=1 ; 
             else
                TrainLabel(i,1)=1;
             end   
        case 6
             if SUM >1800
                 TrainLabel(i,7)=1 ; 
             else
                TrainLabel(i,1)=1;
             end   
        case 7
             if SUM >2100
                 TrainLabel(i,8)=1 ; 
             else
                TrainLabel(i,1)=1;
             end    
        case 8
             if SUM >2400
                 TrainLabel(i,9)=1 ; 
             else
                TrainLabel(i,1)=1;
             end
        case 9
             if SUM >2700
                 TrainLabel(i,10)=1 ; 
             else
                TrainLabel(i,1)=1;
             end
             
        otherwise
            
    end
   
end


%%
A=TrainData(50,3,:);
A=reshape(A,[1,500]);
figure
plot(A)

% Data=TrainData3(1:358,:,:);
% Label=TrainLabel3(1:358,:);
% TrainData=Data;
% TrainLabel=Label;
