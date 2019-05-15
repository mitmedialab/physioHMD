%%
%%%%%% label process %%%%%%
% TS=TestSignal;
% TS=TS(:,8000:end);
% S=size(TS);
% Numb=S(1,2)/500;
% Numb=round(Numb)-1;
% TrainData=zeros(Numb,4,500);
% TrainLabel=zeros(Numb,6);
% OrigData=TrainData;
% filter=fspecial('average',[1,15]); 
% for i=1:2960
% %     % segmentation
% %     TrainData(i,1,:)=TS(4,(500*(i-1)+1):500*i);
% %     TrainData(i,2,:)=TS(5,(500*(i-1)+1):500*i);
% %     TrainData(i,3,:)=TS(10,(500*(i-1)+1):500*i);
% %     TrainData(i,4,:)=TS(11,(500*(i-1)+1):500*i);
% %     
%     % remove the mean
%     M=mean(TrainData(i,:,:),3);
%     TrainData(i,1,:)= TrainData(i,1,:)-M(1);
%     TrainData(i,2,:)= TrainData(i,2,:)-M(2);
%     TrainData(i,3,:)= TrainData(i,3,:)-M(3);
%     TrainData(i,4,:)= TrainData(i,4,:)-M(4);
% %     OrigData(i,:,:)=TrainData(i,:,:);
% %     % remove the trend  
% %     Temp1=reshape(TrainData(i,1,:),[1,500]);
% %     Temp2=reshape(TrainData(i,2,:),[1,500]);
% %     Temp3=reshape(TrainData(i,3,:),[1,500]);
% %     Temp4=reshape(TrainData(i,4,:),[1,500]);
% %     
% %     TrainData(i,1,:)=Temp1-filter2(filter,Temp1);
% %     TrainData(i,2,:)=Temp2-filter2(filter,Temp2 );  
% %     TrainData(i,3,:)=Temp3-filter2(filter,Temp3);  
% %     TrainData(i,4,:)=Temp4-filter2(filter,Temp4);
% % %     A=filter2(filter,Temp4);
% end
 

% %%
% TS=TestSignal;
% TS=TS(:,1:end);
% Temp=TS(11,:);
% filter=fspecial('average',[1,15]); 
% Temp=Temp-filter2(filter,Temp);
% Temp(1,1:50)=0;
% Temp(1,end-50:end)=0;
% 
% S=size(TS);
% Numb=S(1,2)/500;
% Numb=round(Numb)-1;
% TrainData=zeros(Numb,4,500);
% TrainLabel=zeros(Numb,6);
% LabelMark=zeros(1,S(1,2));
% 
% for i=1:11
%     TS(i,:)= mapminmax(TS(i,:),0.1,0.9);
% end

%%  Mark
% Index1=1
% for i=1:Index1
% plot(Temp)
% h=imrect;  
% pos= getPosition(h);
% pos=round(pos);
% LabelMark(1,pos(1):(pos(1)+pos(3)))=2;%% sad
% hold on
% plot(pos(1):(pos(1)+pos(3)),ones(1,pos(3)+1))
% end
% 
% Index2=1
% for i=1:Index2
% plot(Temp)
% h=imrect;  
% pos= getPosition(h);
% LabelMark(1,pos(1):(pos(1)+pos(3)))=1;%% sad
% hold on
% % plot(pos(1):(pos(1)+pos(3)),ones(1,pos(3)+1))
% % end
% 
% for i=1:Numb
%     % segmentation
%     TrainData(i,1,:)=TS(4,(500*(i-1)+1):500*i);
%     TrainData(i,2,:)=TS(5,(500*(i-1)+1):500*i);
%     TrainData(i,3,:)=TS(10,(500*(i-1)+1):500*i);
%     TrainData(i,4,:)=TS(11,(500*(i-1)+1):500*i);   
%     Marker(i,:)=LabelMark(1,(500*(i-1)+1):500*i); 
%     % remove the mean
% %     M=mean(TrainData(i,:,:),3);
% %     TrainData(i,1,:)= TrainData(i,1,:)-M(1);
% %     TrainData(i,2,:)= TrainData(i,2,:)-M(2);
% %     TrainData(i,3,:)= TrainData(i,3,:)-M(3);
% %     TrainData(i,4,:)= TrainData(i,4,:)-M(4);
% end


% close all
% for i=135:145
%         Temp=TrainData(i,4,:);
%         TempTS=TS(11,1:end);
% %         TempTS=TS(5,60000:end);
%         TempTS=TempTS-mean(TempTS(:));
%         TempTS=TempTS-filter2(filter,TempTS);  
%         figure
%         plot(1:113285,TempTS);
% %         plot(60000:128000,TempTS);
%         hold on
%         plot((500*(i-1)+1):500*i,Temp(:));
% end

% clear
% AAA=TS(10,30000:31000);
% AAA=AAA-mean(AAA);
% % g=TS(10,:);
% g=AAA;
% A=fspecial('average',[1,10]); 
% Y=filter2(A,g);         
% plot(Y)
% B=AAA-Y;
% plot(B)
% hold on
% plot(AAA)

% plot(TS(4,:))
% hold on
TS(5,:)=medfilt2(TS(5,:),[1,3]);
plot(TS(5,:))
%%
%ÖØÐÂÅÅÐò
S=size(TrainData);
Index=randperm(S(1));
TrainSetTemp=TrainData;
TrainSetLabelTemp=TrainLabel;
% OrigDataTemp=TrainData;
for i=1:S(1)
TrainSetTemp(i,:,:)=TrainData(Index(i),:,:);
TrainSetLabelTemp(i,:)=TrainLabel(Index(i),:); 
% OrigDataTemp(i,:,:)=OrigData(Index(i),:,:);
end
TrainData=TrainSetTemp;
TrainLabel=TrainSetLabelTemp;
% clear TrainSetLabelTemp  TrainSetTemp
% OrigData=OrigDataTemp;
A=TrainSetTemp(100,:,:);
A=reshape(A,[4,500]);
plot(A(1,:))


%% Ðý×ª
for i=1:4633
    
    A1=reshape(TrainData(i,1,:),[1,500]);    
    A2=reshape(TrainData(i,2,:),[1,500]); 
    A3=reshape(TrainData(i,3,:),[1,500]); 
    A4=reshape(TrainData(i,4,:),[1,500]); 
   
    TrainData1(i,1,:)=rot90(A1,2);    
    TrainData1(i,2,:)=rot90(A2,2); 
    TrainData1(i,3,:)=rot90(A3,2); 
    TrainData1(i,4,:)=rot90(A4,2); 
 
end
TrainData=[TrainData;TrainData1];
TrainLabel=[TrainLabel;TrainLabel];
