function SigAnly(S)
% Wavelet transform
[LoF,HiF] = dwt(S,'db1');
HiF_abs=abs(HiF);
LoF_abs=abs(LoF);
temp1= wextend('1D','sp0',HiF_abs,50,'b');
temp2= wextend('1D','sp0',LoF_abs,50,'b');
filter_Hi=fspecial('gaussian',[1,150],11);
temp1=imfilter(temp1,filter_Hi); 
temp2=imfilter(temp2,filter_Hi);
HiF_abs=temp1(50:50+4999);
LoF_abs=temp2(50:50+4999);
clear temp1  temp2
LoF_diff=diff(LoF_abs);
LoF_diff=LoF_diff-mean(LoF_diff(:));
LoF_diff=medfilt2(LoF_diff,[1,5]);
figure
subplot(2, 2, 1); plot(S); title('Original Signal');
subplot(2, 2, 2); plot(HiF_abs, 'r-'); title('Hig Frenqunecy Analysis');
subplot(2, 2, 3); plot(LoF_abs); title('Low Frenqunecy Analysis1');
subplot(2, 2, 4); plot(LoF_diff); title('Low Frenqunecy Analysis2');
end