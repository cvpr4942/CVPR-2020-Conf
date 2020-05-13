function inds=SP(A,K,maxiter)

%%%%% The main selection code for "Select to Better Learn: 
      %Fast and Accurate Deep Learning using Data Selection from Nonlinear Manifolds" 
      %Published in CVPR 2020

%The code is written by Mohsen Joneidi (moh872@yahoo.com)

%Input: A is data or kernel 
%	K is the number desired samples (nodes)
%	maxiter is an integer less than 10
%       maxiter = 1 and inds=[] results in IPM algorithm published in CVPR 2019
%       maxiter = 5 is fine for convergence

%Output: indices of selected samples (nodes)

inds=randperm(size(A,2),K);
for i=1:maxiter
    for k=1:K
    A_k=A(:,setdiff(inds,inds(k))); %all selected data except kth
    Ek=A-A_k*pinv(A_k)*A;% project all data on null-space of A_k
    [u,~]=eigs(Ek*Ek',1);% the most critical direction that K-1 samples cannot cover
    cr=Ek'*u;
    [~,p]=max(abs(cr)); 
    inds(k)=p; %kth selected data tells the rest K-1 that "dont worry I try to cover your null" 
    end
end
end