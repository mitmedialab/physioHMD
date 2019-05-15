clear all
clc
load('smile_02.mat');
ch = 11 ;
Ns = 1000 ;
w0_pwl = 2*60/Ns ;
w1_pwl = 2*120/Ns ;
w2_pwl = 2*180/Ns ;
w3_pwl = 2*240/Ns ;
w4_pwl = 2*300/Ns ;
w5_pwl = 2*360/Ns ;
w6_pwl = 2*420/Ns ;
bw_pwl = 2*10/Ns ;
wp_emg = 2*5/Ns;
ws_emg = 2*50/Ns;
C(ch,:) = C(ch,:) - mean(C(ch,:));
C(ch,:) = C(ch,:)/max(C(ch,:));
[a_pwl,b_pwl] = iirnotch(w0_pwl,bw_pwl);
[a1_pwl,b1_pwl] = iirnotch(w1_pwl,bw_pwl);
[a2_pwl,b2_pwl] = iirnotch(w2_pwl,bw_pwl);
[a3_pwl,b3_pwl] = iirnotch(w3_pwl,bw_pwl);
[a4_pwl,b4_pwl] = iirnotch(w4_pwl,bw_pwl);
[a5_pwl,b5_pwl] = iirnotch(w5_pwl,bw_pwl);
[a6_pwl,b6_pwl] = iirnotch(w6_pwl,bw_pwl);
[a_emg,b_emg] = butter(2, ws_emg,'high');
[a_chk,b_chk] = butter(3,[wp_emg,ws_emg],'bandpass');
temp1 = filter(a_pwl,b_pwl,C(ch,:));
temp = filter(a_chk,b_chk,temp1);
temp1 = filter(a1_pwl,b1_pwl,temp1);
temp1 = filter(a2_pwl,b2_pwl,temp1);
temp1 = filter(a3_pwl,b3_pwl,temp1);
temp1 = filter(a4_pwl,b4_pwl,temp1);
temp1 = filter(a5_pwl,b5_pwl,temp1);
temp1 = filter(a6_pwl,b6_pwl,temp1);
%temp = filter(a_emg,b_emg,temp1);
subplot(3,1,1)
plot(C(ch,:));
subplot(3,1,2)
plot(temp1);
subplot(3,1,3)
plot(temp);
