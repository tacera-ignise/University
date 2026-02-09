function [x] = Protsvitaiev_ip(A, B, dokladnosc, X0)    
    n = length(B);
    x = X0; % Przybliżone rozwiązanie początkowe
    max_iter = 1000; % Maksymalna liczba iteracji (możesz dostosować)
    iter = 0;
    
    while iter < max_iter
        x_new = x;
        
        for i = 1:n
            sigma = A(i, :) * x_new - A(i, i) * x_new(i);
            x(i) = (B(i) - sigma) / A(i, i);
        end
        
        % Sprawdzenie warunku zbieżności
        if norm(x - x_new, inf) < dokladnosc
            break;
        end
        
        iter = iter + 1;
    end
    
    if iter == max_iter
        disp('Układ równań może nie być zbieżny w danej liczbie iteracji.');
    else
        disp(['Rozwiązanie znalezione po ', num2str(iter), ' iteracjach.']);
    end
end
