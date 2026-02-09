[X,Y]=meshgrid([-10:0.5:10],...
               [-10:0.5:10]);
f=@(x,y) x.^2+y.^2;
subplot(2,1,1)
mesh(X,Y,f(X,Y))
subplot(2,1,2)
contour(X,Y,f(X,Y))
axis square
dx=[-5,-2,-2,0]
dy=[-3,-3,0,0]
hold on
plot(dx(1),dy(1),"r*")
plot(dx(end),dy(end),"r*")
plot(dx(2:end-1),...
    dy(2:end-1),'g.')

for i=1 : numel(dx)-1
    line([dx(i),dx(i+1)],[dy(i),dy(i+1)],'LineWidth',2,'Color','red');
end