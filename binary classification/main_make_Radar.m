% written by Renjia, 
% Zhejiang Sci-Tech University, Hangzhou, China
% 2021-12-30

% draw radar picture

clear;
close all;
clc;
tic
load mic;        
[mic1,order]=sort(abs(mic(1,:)),'descend');
t = 0:pi/26:2*pi; 
a = load('d00.dat');     %load the training data      %52*500
a = a';                                    % 500*52;
mina = min(a);
maxa= max(a);
% a = (a-mina)./(maxa-mina);
% a = a(:,order);
% a(:, end+1)=a(:,1);
% for i=1:size(a,1)
%     K = a(i,:)';
%     y = diag(K*sin(t));
%     x = diag(K*cos(t));
%     fig = figure(i);
%     set(fig,'visible','off')
%     fill(x',y','black');
%     axis equal
%     axis off; %
%     print(fig, '-dpng',strcat('D:\radar_graph3\d00_',num2str(i),'.png')); %draw the radar graph of normal training data
% end

for index1 = 1:21   % index: fault no.
     fault = sprintf('d%02i',index1);   
     name_fault_train = strcat(fault,'.dat');      
     name_fault_test = strcat(fault,'_te.dat');
     b=load(name_fault_train);    % load the training data of fault 480*52
     c=load(name_fault_test);     % load the test data of fault 960*52
     b = (b-mina)./(maxa-mina);
     c = (c-mina)./(maxa-mina);
     b = b( :,order);
     c = c( :, order);
     b(:,end+1)=b(:,1);
    for i=1:size(b,1)
        K = b(i,:)';
        y = diag(K*sin(t));
        x = diag(K*cos(t));
        fig = figure(i);
        set(fig,'visible','off')
        fill(x',y','black');
        axis equal
        axis off;% 
        print(fig, '-dpng',strcat('D:\radar_graph3\',fault,'_',num2str(i),'.png')); %
    end
    c(:,end+1)=c(:,1);
    for i=1:size(c,1)
        K = c(i,:)';
        y = diag(K*sin(t));
        x = diag(K*cos(t));
        fig = figure(i);
        set(fig,'visible','off')
        fill(x',y','black');
        axis equal
        axis off;
        print(fig, '-dpng',strcat('D:\radar_graph3\',fault,'_te_',num2str(i),'.png')); 
    end
    
end    

toc

