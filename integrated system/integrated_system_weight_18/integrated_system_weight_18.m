clear;
close all;
clc;
tic

testno = 10;  
T = 40;
Len1 = 480-T+1;
Len2 = 800-T+1;
filename = strcat('inte_weight_18.xls');

load mic;
[~,order]=sort(abs(mic(1,:)),'descend');                                                                                                                                                                                                                                                                   

[fault_train, fault_test]=load_normalize( );

for index = 1:21
    testy ((index-1)*Len2+1:(index-1)*Len2+Len2,:)= index;
end

%Remove faults 3, 9, and 15
A = [3,8,13];
for index1 = A
   testy((index1-1)*Len2+1:(index1-1)*Len2+Len2,:) = [];
end
%turn label into catogory
testy=categorical(testy);


for i = 1:testno

    %FFT
    py1 = FFT_weight_18(fault_train, fault_test,order);
    py_1 = zeros(Len2*18,18);
    for m1 = 1:Len2*18
        y = squeeze(py1(:,:,:,m1));
       py_1(m1,:) = y'; 
    end


    %T*1*F
    py2 = T1F_weight_18(fault_train, fault_test,order);
    py_2 = zeros(Len2*18,18);
    for m2 = 1:Len2*18
        y = squeeze(py2(:,:,:,m2));
       py_2(m2,:) = y'; 
    end

    %T*F*1
    py3 = TF1_weight_18(fault_train, fault_test,order);
    py_3 = zeros(Len2*18,18);
    for m3 = 1:Len2*18
        y = squeeze(py3(:,:,:,m3));
       py_3(m3,:) = y'; 
    end

    %WT
    py4 = WT_weight_18(fault_train, fault_test,order);
    py_4 = zeros(Len2*18,18);
    for m4 = 1:Len2*18
        y = squeeze(py4(:,:,:,m4));
       py_4(m4,:) = y'; 
    end

    %From top to bottom, FFT, T*1*F, T*F*1, WT identify the confidence of each fault
    W = [0.998 	1.000   0.988 	0.726 	0.947 	0.986 	0.788   0.503 	0.984 	0.851 	0.654 	1.000   0.548 	0.996 	0.810 	0.997 	0.813 	0.797;
         0.925 	0.974   0.965 	0.855 	0.659 	0.969 	0.668   0.695 	0.881 	0.888 	0.428 	0.989   0.639 	0.929 	0.836 	0.561 	0.908 	0.072;
         0.999 	0.994   0.987 	0.784 	0.981 	0.995 	0.703   0.366 	0.951 	0.711 	0.171 	1.000   0.355 	0.932 	0.862 	0.984 	0.871 	0.193;
         0.997 	0.986   0.993 	0.773 	0.963 	0.995 	0.715   0.312 	0.948 	0.698 	0.200 	0.999   0.296 	0.980 	0.873 	0.967 	0.883 	0.119];

     %W_w is the confidence coefficient
     x = 3;  %x can take integers such as 1, 2, 3, 4, etc., indicating multiples
     W_w = [1  x  1  1  1  1  x  1  x  1  x  x  1  x  1  x  1  x;
            1  1  1  x  1  1  1  x  1  x  1  1  x  1  1  1  x  1;
            x  1  1  1  x  x  1  1  1  1  1  x  1  1  1  1  1  1;
            1  1  x  1  1  x  1  1  1  1  1  1  1  1  x  1  1  1];
     W = W.*W_w;
     py_s = zeros(4,18);
     py = zeros(Len2*18,1);
     weight = zeros(1,4);
   
      %weight
      for n = 1:Len2*18
        py_s(1,:) = py_1(n,:);
        py_s(2,:) = py_2(n,:);
        py_s(3,:) = py_3(n,:);
        py_s(4,:) = py_4(n,:);
        py_middle = W' * py_s;
        [~,j] = max(diag(py_middle));
        % Restore labels
        if j<=2
            j1 = j;
        elseif (j>=3 && j<=7)
            j1 = j+1;
        elseif (j>=8 && j<=12)
            j1 = j+2;
        else
            j1 = j+3;
        end
        py(n,:) = j1;
      end
      
     py = categorical(py);
     [cm,~] = confusionmat(testy,py);
     
     precision = diag(cm)./sum(cm,2);
     recall = diag(cm)./sum(cm,1)';
     f1 = 2*precision.*recall./(precision+recall+0.00001);% macro F1score

    %     write the data into excel file
    xlslocation = strcat('B',num2str(i+3),':V',num2str(i+3));
    xlslocation1 = strcat('B',num2str(i+14),':V',num2str(i+14));
    xlslocation2 = strcat('B',num2str(i+25),':S',num2str(i+25));
    xlswrite(filename,precision',1,xlslocation);
    xlswrite(filename,recall',1,xlslocation1);
    xlswrite(filename,f1',1,xlslocation2);
    
end
toc

 
