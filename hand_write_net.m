p = -2:0.01:2;
f = @(x)sin(pi*x/4)+1;
plot(p,f(p));
hold on
w1=[-0.27 -0.41]';
w2=[0.09 -0.17]';
b1=[-0.48 -0.13]';
b2=0.48;
s=@(x)1./(1+exp(-x));
% plot(p,s(p));
a=[0 0]';
n2=0;
n1=[0 0]';
x_train=[-1 -1 0 1 2]';
y_train=f(x_train);
yita=0.1;
while (1)
    for j=1:length(x_train)
      %%  ?????¡è???¨®??
       n1(:,j)=x_train(j)*w1(:,j)+b1(:,j);
       a(:,j)=s(n1(:,j));
       n2(j)=w2(:,j)'*a(:,j)+b2(j);
       error(j)=y_train(j)-n2(j);
       %%   ¡¤??¨°?¨¹??????
       b2(j+1)=b2(j)+yita*error(j);
       w2(:,j+1)=w2(:,j)+yita*error(j)*a(:,j);
       b1(:,j+1)=b1(:,j)+yita*error(j).*w2(:,j).*a(:,j).*(1-a(:,j));
       w1(:,j+1)=w1(:,j)+yita*error(j)*w2(:,j).*a(:,j).*(1-a(:,j))*x_train(j);
    end
    b2(1)=b2(j+1);
    w2(:,1)=w2(:,j+1);
    b1(:,1)=b1(:,j+1);
    w1(:,1)=w1(:,j+1);
    if(norm(w1(:,j)-w1(:,j+1))+norm(w2(:,j)-w2(:,j+1))<0.0001)
        break;
    end
end
beita1=s(p*w1(1,1)+b1(1,1))*w2(1,1);
beita2=s(p*w1(2,1)+b1(2,1))*w2(2,1);
y_model=beita1+beita2+b2(1);
plot(p,y_model);
legend("true","model")
