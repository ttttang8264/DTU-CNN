function [fault_train, fault_test]=load_normalize( )
% load and normalize the data
a=load('d00.dat');     %load the training data
a=a';                           %500*52
maxa = max(a);
mina = min(a);

for i = 1:21   % index: fault no.    
    name_fault_train = strcat(sprintf('d%02i',i),'.dat');      
    name_fault_test = strcat(sprintf('d%02i',i),'_te.dat');
    fault_train{i}=load(name_fault_train) ;    % load the training data of fault 480*52
    fault_train{i} =( fault_train{i} - mina) ./ (maxa-mina); % normalize the faulty training data
    temp_test = load(name_fault_test);  
    fault_test{i}= temp_test (161:end,:);  % load the test data of fault 960*52, get the last 800 data for test
    fault_test{i} =( fault_test{i} - mina) ./ (maxa-mina);   % normalize the faulty test data
end
end
