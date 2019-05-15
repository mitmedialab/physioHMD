function y=highPass(x)
[b,a]=butter(3,0.095,'high');
y=filter(b,a,x);
% subplot 211; plot(x(20:end)); title('Original Signal');
% subplot 212; plot(y(20:end)); title('Output of High-pass filter');
end