close
[x,y]=meshgrid([-20:0.5:20],[-20:0.5:20]);
%f=@(x,y) x.^2+y.^2;
%f=@(x,y) (5*(x-1).^2+10*(y-3).^2);
%f=@(x,y) -cos(x).*cos(y).*exp(-((x-pi).^2+(y-pi).^2))
f= @(x,y) 100*(y-x.^2).^2+(1-x).^2
%{
i=34 10 10
i=12 -19 -19
i=28    4 4
%}
plot3(x,y,f(x,y));
x0=-2
y0=-1
k=2.7
doc=0.00001
dx=[x0]
dy=[y0]
i=0
tic
%Metoda 1

while k>doc
    T=[x0,x0+k,x0,x0-k,x0;
       y0,y0,y0-k,y0,y0+k;
       f(x0,y0),f(x0+k,y0),f(x0,y0-k),f(x0-k,y0),f(x0,y0+k)]
    [w,p]=min(T(3,:));
    if p==1
       k=k/3
    end
    if p==2
       x0=x0+k
       
    end
    if p==3
        y0=y0-k
        
    end
    if p==4
       x0=x0-k
       
    end
    if p==5
       y0=y0+k
       
    end
    dx=[dx,x0]
    dy=[dy,y0]
    i=i+1
end
czas=toc
%metoda 2
%{
while true
        iter = iter + 1;
        
        gradient_x = (f(x + krok, y) - f(x - krok, y)) / (2 * krok);
        gradient_y = (f(x, y + krok) - f(x, y - krok)) / (2 * krok);
        
        x = x - krok * gradient_x * randi(4);
        y = y - krok * gradient_y * randi(4);
        
        sciezka = [sciezka; x, y];
        
        if abs(gradient_x) < dok && abs(gradient_y) < dok
            break;
        end
end

%}



%{
vec=[]
vec_x=[x0-k]
vec_y=[y0+k]

while k>doc
    for j=1:89
        y0=y0+(sin(k))
        x0=x0-(k-cos(k))
        vec_x=[vec_x,x0]
        vec_y=[vec_y,y0]
    end
    for j=1:90
        y0=y0-(sin(k))
        x0=x0+(cos(k))
        vec_x=[vec_x,x0]
        vec_y=[vec_y,y0]
    end
    for j=1:90
        y0=y0-(sin(k))
        x0=x0-(cos(k))
        vec_x=[vec_x,x0]
        vec_y=[vec_y,y0]
    end
    for j=1:89
        y0=y0+(sin(k))
        x0=x0+(cos(k))
        vec_x=[vec_x,x0]
        vec_y=[vec_y,y0]
    end
    
    vec=[f(vec_x(:),vec_y(:))]
    [w,p]=min(vec)
    dx=[dx,x0]
    dy=[dy,y0]
    i=i+1
end
%}
  %{
    while true
        iter = iter + 1;
        gradient_x0 = (f(x0 + k, y0) - f(x0 - k, y0)) / (2 * k);
        gradient_y0 = (f(x0, y0 + k) - f(x0, y0 - k)) / (2 * k);
        
        kat = rand() * 360;
        x0 = x0 - k * gradient_x0 * cosd(kat);
        y0 = y0 - k * gradient_y0 * sind(kat);
        
        dx = [dx, x0];
        dy = [dy, y0];
        
        
        if abs(gradient_x0) < dok && abs(gradient_y0) < dok
            break;
        end
    end
  %}



subplot(2,1,1)
mesh(x,y,f(x,y));
subplot(2,1,2)
contour(x,y,f(x,y));
hold on;
plot(dx(1),dy(1),'r*')
plot(dx(end),dy(end),'r*')
plot(dx(2:end-1),dy(2:end-1),'g*')
for i=1:numel(dx)-1
    line([dx(i),dx(i+1)],[dy(i),dy(i+1)],'LineWidth',2,'Color','Magenta')
end

%{

%}