%--------------------------Metodę połowienia,Metodę Newtona
clc
close
clear all
%fun = @(x)(((x.^2)/3)-(13*x/7)+11);
fun=@(x) (x.^2)
%fun = @(x)((x-3).^3/3)+(x-4).^4
x0=1
df = @(x,h) (fun(x+h)-fun(x-h))/(2*h);
d2f = @(x,h) (fun(x+h) - (2*fun(x)) + fun(x-h))/h^2; 
h = 1/2^3;
dok = 10^(-2);
%[x,i]=pol(a,b,fun,dok)
%{
Metoda połowienia
fun=@(x) x.^2
Dokladność  10^(-2)
przedział:  <-3, 7>     <0, 10>     <-10, 0>        <-500, 500>
wynik:      8.5831e-06  2.3842e-05  2.3842e-05      0
iteracja:   10          10          10              17

Dokladność  10^(-5)
przedział:  <-3, 7>     <0, 10>     <-10, 0>        <-500, 500>
wynik:      3.6380e-12  2.2737e-11  2.2737e-11      0
iteracja:   10          20          20              27

/////////////////////////////////////////////

fun = @((x^2)/3)-((13*x)/7)+11
Dokladność  10^(-2)
przedział:  <-3, 7>     <0, 10>     <-10, 0>        <-500, 500>
wynik:      8.4133      8.4133      11.0091         8.4133
iteracja:   10          10          10              17

Dokladność  10^(-5)
przedział:  <-3, 7>     <0, 10>     <-10, 0>        <-500, 500>
wynik:      8.4133      8.4133      11.0000         8.4133
iteracja:   20          20          20              27

///////////////////////////////////////////

fun = @(x)((x-3).^3/3)+(x-4).^4
Dokladność  10^(-2)
przedział:  <-3, 7>     <0, 10>     <-10, 0>        <-500, 500>
wynik:      0.0959      0.1470      248.2083        0.0959
iteracja:   10          10          10              17

Dokladność  10^(-5)
przedział:  <-3, 7>     <0, 10>     <-10, 0>        <-500, 500>
wynik:      0.0959      0.1445      247.0012        0.0959
iteracja:   27          20          20              27
%}

a = -500; %lewa granica przedziału
b = 500;  %prawa granica przedziału
%[x,i]=Newton(a,b,fun,dok,df,d2f,h)
%{
Metoda Newton
fun=@(x) x.^2
Dokladność  10^(-2)
przedział:  <-3, 7>     <0, 10>     <-10, 0>        <-500, 500>
wynik:      0           0           0               0
iteracja:   1           1           1               1

/////////////////////////////////////////////

fun = @((x^2)/3)-((13*x)/7)+11
Dokladność  10^(-2)
przedział:  <-3, 7>     <0, 10>     <-10, 0>        <-500, 500>
wynik:      8.4133      8.4133      8.4133          8.4133
iteracja:   3           3           3               3

///////////////////////////////////////////

fun = @(x)((x-3).^3/3)+(x-4).^4
Dokladność  10^(-2)
przedział:  <-3, 7>     <0, 10>     <-10, 0>        <-500, 500>
wynik:      0.0960      0.0960      0.0960          0.0960
iteracja:   14          14          14              27
  
%}

%{
Pytania:
1)Metodę Newtona wykonuje najmniejszą liczbę iteracji.
2)Funkcja powinna być ciągla na tym przedziale
3) Nie wpływa
4)-
$}

disp('iteracje')
disp(i) 
disp("min wartosc funkcji")
disp(fun(x(end)))
figure
plot(fun(x),'rd') 
axis([1,i, min(fun(x)), max(fun(x))]) %dopasowanie zakresów osi
xlabel('Iteracje')
ylabel('Wartość funkcji')
title(['Wartość funkcji w x_min',num2str(fun(x(end)))]),
options = optimset('PlotFcns',@optimplotfval);
x_test = fminsearch(fun,x0, options)