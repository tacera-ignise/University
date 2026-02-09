%skrypt do zadania "Metoda graficzna w programowaniu liniowym"
clc
close
clear

%dane pochodzące z treści zadania - tylko tych informacji wymagamy od
%użytkownika, reszta kodu pozostaje dla niego 'czarną magią'

% A=  % macierz współczynników z ograniczeń
% B=  % wektor wyrazów wolnych z ograniczeń   
% Z=  % wektor przechowujący informację o znaku nierówności, np. 1 dla > i >=, 0 dla =, -1 dla < i <=         
% F=  % wektor współczynników z funkcji celu 
%dla danych z zadania z wykładu:

A = [3 5;
     5 2;
     1 0;
     0 1];
B = [15;
     10;
     0;   
     0];
Z = [-1;
       -1;
        1;
        1];
F = [5, 3];

%-----------------
% wszystko od tego miejsca, to obliczenia na podstawie wprowadzonych
% informacji
%--- opcja
% opcja - do zrobienia na końcu, jak będzie działała wersja podstawowa 
% należy sprawdzić czy w wektorze B są same dodatnie wartości, jeśli nie co
% trzeba zrobić? tu jest na to miejsce.
%-----------

% przydaje się zmienna przechowująca informację o liczbie zmiennych w
% zadaniu. Wartość tej zmiennej można określić na kilka różnych sposobów, np.:
[ogr, zm] = size(A) %co, w czy k, będzie informowało o licznie zmiennych?
% albo
%n = numel(...) % do którego wektora sięgnąć?
    
%poszukujemy wszystkich rozwiązań par ograniczeń, w tym celu konieczne jest
%zbudowanie pętli (jednej lub więcej), które pozwolą nam rozwiązywać
%następujące układy równań liniowych: pierwsze ograniczenie z drugim,
%pierwsze z trzeciem, pierwsze z czwartym, itd., potem drugie z trzeciem,
%drugie z czwartym, itd., należy sprawdzić wszystkie kombinacje w przód bez
%powtórzeń. Rozwiązaniem układu składającego się z dwóch ograniczeń (dlaczego z dwóch i czy zawsze z dwóch?) będzie pionowy wektor dwuelementowy, którego
%elementami są współrzędne punktu przecięcia się tych dwóch ograniczeń.
%Jeśli elementy wektora są rzeczywiste, to taki wektor zapisujemy do
%jakiejś tablicy. Do wykrywania elementów innych niż rzeczywiste służą
%polecenia isinf i isnan - można (a nawet trzeba) z nich skorzystać.
k=1
for i=1:ogr-1
    AT(1,:)=A(i,:)
    BT(1,:)=B(i,:)
    for j=i+1:ogr
        AT(2,:)=A(j,:)
        BT(2,:)=B(j,:)
        w=AT\BT;
        if sum(isinf(w))==0
           R(:,k)=w;
           k=k+1;
       end
    end
end


%--------------> 

% najtrudniejszy etap - wydobycie ze wszystkich rozwiązań tylko rozwiązań
% dopuszczalnych. Jakie rozwiązanie nazywamy dopuszczalnym? Rozwiązania
% dopuszczalne gromadzimy w innej (nowej) tablicy.

u=1
for i=1:k-1
    flaga=0
    for j=1:ogr
        if Z(j)==1
            if A(j,1)*R(1,i)+A(j,2)*R(2,i)>=B(j)
                flaga=flaga+1
            end
        end
        if Z(j)==-1
            if A(j,1)*R(1,i)+A(j,2)*R(2,i)<=B(j)
                flaga=flaga+1
            end
        end
        if Z(j)==0
            if A(j,1)*R(1,i)+A(j,2)*R(2,i)==B(j)
                flaga=flaga+1
            end
        end
    end
    if flaga==ogr
        Wynik(:,u)=R(:,i)
        u=u+1

    end
end


%-------------->
       
% rysujemy ograniczenia - jak przy pomocy polecenia plot narysować równanie
% prostej o określonych współczynnikach (skąd wziąć te współczynniki)?

 

%-------------->

 
% kolorujemy obszar rozwiązań dopuszczalnych, należy użyć pary poleceń:

hold on;
K = convhull(Wynik')
fill(Wynik(1,K),Wynik(2,K), 'rd');
x = min(R(1,:))-1:0.01:max(R(1,:))+1;
y=min(R(2,:))-1:0.01:max(R(2,:))+1

for i=1:ogr
    if (A(i,2) == 0)
        x1=(B(i)/A(i,1))*ones(size(y))
        plot(x1,y);
    else
        
        y1 = (B(i)-A(i,1)*x)/A(i,2);

        plot(x,y1);
    end
end


% obliczamy dla którego rozwiązania dopuszczalnego funkcja celu przyjmuje
% wartość największą

max_1 = 0;
for i=1:zm
    max_next = F(1,1)*Wynik(1,i)+F(1,2)*Wynik(2,i);
    if (max_next>max_1)
        max_1 = max_next;
    end
end
max_1

% rysujemy (polecenie plot) funkcję celu tak aby przechodziła przez punkt, w którym
% przyjmuje największą wartość 
x = min(R(1,:))-1:0.01:max(R(1,:))+1;
y=(max_1-F(1)*x)/F(2)
plot(x,y)

% --


% wykres otrzymuje tytuł
%--
title(['Maksymalna wartość funkcji celu wynosi:', num2str(max_1)])