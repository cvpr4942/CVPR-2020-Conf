def SP(data,K):
  A=data
  At=data
  inds=np.zeros(K,);
  inds=inds.astype(int)
  iter=0
  for k in range(0,K):
    iter=iter+1
    [U,S,V]=np.linalg.svd(At,full_matrices=False)
    u=U[:,1]
    v=V[:,1]
    N=np.linalg.norm(At,axis=0)
    B=At/N
    B=np.transpose(B)
    Cr=np.abs(np.matmul(B,u))
    ind=np.argsort(Cr)[::-1]
    p=ind[0]
    inds[k]=p
    A3=A[:,inds[0:k+1]]
    At=A-np.matmul(np.matmul(A3,np.linalg.pinv(np.matmul(np.transpose(A3),A3))),np.matmul(np.transpose(A3),A))
   ind2=np.zeros(K-1,);
   for iter in range(1,5):
     for k in range(0,K):
       ind2=np.delete(inds,k)
       A3=A[:,ind2]
       At=A-np.matmul(np.matmul(A3,np.linalg.pinv(np.matmul(np.transpose(A3),A3))),np.matmul(np.transpose(A3),A))
       [U,S,V]=np.linalg.svd(At,full_matrices=False)
       u=U[:,1]
       v=V[:,1]
       N=np.linalg.norm(At,axis=0)
       B=At/N
       B=np.transpose(B)
       Cr=np.abs(np.matmul(B,u))
       ind=np.argsort(Cr)[::-1]
       p=ind[0]
       inds[k]=p
  return inds