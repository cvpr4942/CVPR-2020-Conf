function inds=SP(A,K)
   maxiter= 1;
   At=A;
for  k=1:K
    [~,s,v]=svds((At),1);
    cr=s*abs(v);
    [~,SET]=sort(cr,'descend');  
    p=SET(1);
    inds(k)=p;
    A3=A(:,inds);
    At=A-A3*inv(A3'*A3)*A3'*A;
end

for i=1:maxiter-1
    for k=1:K
    A3=A(:,setdiff(inds,inds(k)));
    At=A-A3*inv(A3'*A3)*A3'*A;
    [~,s,v]=svds((At),1);
    cr=s*abs(v);
    [~,p]=max(cr);
    inds(k)=p;
    end
end
end