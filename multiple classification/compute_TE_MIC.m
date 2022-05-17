a=load('d00.dat');     %Requires a 52x500 matrix
minestats=zeros(52,52);
for i=1:52
    for j=1:52
        if(i<=j)
            m=mine(a(i,:),a(j,:));
            minestats(i,j)=m.mic;
        end
    end
end
for i=1:52
    for j=1:52
        if(i>j)
            minestats(i,j)=minestats(j,i);
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Here's how to fully compute the matrix at the beginning
%But it is observed that the MIC matrix is symmetrical about the diagonal, so the above method is used.
%This method saves half the computation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i=1:52
%     for j=1:52
%         m=mine(a(i,:),a(j,:));
%         minestats(i,j)=m.mic;
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save minestats.mat