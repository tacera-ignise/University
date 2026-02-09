clc
close
clear all
%{
xg = [-10,-6.2, -1.2, -0.9, -0.4, 0.4, 0.9, 1.2, 6.2, 10];
yg = [0, 4.5, 1.8, 4, 2.8, 2.8, 4, 1.8, 4.5, 0];
 
xgg = min(xg):0.01:max(xg);
ygg = min(yg):0.01:max(yg);
 
xd = [-10, -8, -5, -2, 0, 2, 5, 8, 10];
yd =  [0, 0.3, -1.4, -1.6, -3.6, -1.6, -1.4, 0.3 0];
 
ydg = min(yd):0.01:max(yd);
xdg = min(xd):0.01:max(xd);
 
hold on;
 
plot(xg,yg, 'r*');
plot(xd,yd, 'r*');
 
zg = interp1(xg,yg,xgg,'linear');
plot(xgg, zg);
zg = interp1(xd,yd,xdg,'linear');
plot(xdg, zg);
%}

% Metoda Monte Carlo  
%{

N = 500; 
a = min(xg);
b = max(xg);
q = min(yd);
w = max(yg);

celne = 0;
 
hold on;
 
for i = 1:N
    xl = rand() .* (b-a) + a;
    yig = interp1(xg,yg,xl,'linear');
    yid = interp1(xd,yd,xl,'linear');
    yl = rand() .* (w-q) + q;
    if (yl < yig && yl > yid)
        celne = celne + 1;
        plot(xl,yl, 'rX');
    else
        plot(xl, yl, 'gX');
    end
end
 
calka = (celne / N) * (b-a) * (w-q)
%}

% Metoda Protstokąta
%{
s=0
dx=0.1
if (xg(1)~=xd(1))||(xg(end)~=xd(end))||(yg(1)~=yd(1))||(yg(end)~=yd(end))
    return
end
x1=xg(1);
while x1+dx<=xg(end)
    y_g=interp1(xg,yg,x1,"linear");
    y_d=interp1(xd,yd,x1,"linear");
 
    if sign(y_g)==sign(y_d)
        s=s+dx*abs(y_g-y_d);
        line([x1,x1],[y_g,y_d])
        line([x1+dx,x1+dx],[y_g,y_d])
    else
        s=s+dx*(abs(y_g)+abs(y_d));
        line([x1,x1],[0,y_g])
        line([x1+dx,x1+dx],[0,y_g])
        line([x1,x1],[0,y_d])
        line([x1+dx,x1+dx],[0,y_d])
    end
 
    line([x1,x1+dx],[y_g,y_g])
    line([x1,x1+dx],[y_d,y_d])
    x1=x1+dx;
end
%}

%Algorytm całkowania metodą trapezów
%{
f=@(x)sin(x)
s=0
n=1000
xp=-1
xk=1
dx=(xk-xp)/n
i=0
 while i<=n
    if i <n
        s=s+f(xp+i*dx)
    else
        s=(s+(f(xp)+f(xk))/2)*dx

    end
    i=i+1
 end
calka=s
%}
%{
Algorytm całkowania metodą trapezów
Wartość korku   10          100         500         1000
Wynik:          -0.1683     -0.0168     -0.0034     -0.0017
Wielkość kroku całkowania wpływa na wynik .
Powiekszanie kroków ulepsza wynik.
%}


% Metoda Monte Carlo  
f=@(x)sin(x)
N = 1000; 
a = -1;
b = 1;
calka = 0;
dx=a-b
i=0
for i = 1:N
    xl = rand() .* (b-a) + a;
    calka=calka+ f(xl)
    i=i+1
    
end
calka = dx*(calka/N)
 
%{
Metoda Monte Carlo
Wartość korku   10          100         500         1000
Wynik:          -0.2434      0.1780     -0.0109     0.0565
Liczba strzałów  nie wpływa na wynik.

%}