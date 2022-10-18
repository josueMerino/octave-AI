map = imread("mapa.jpg");

dimensions = size(map)

m = (dimensions(2)/dimensions(1));

for i=1:dimensions(1)
    y = m * i;
    y = ceil(y);
    map(i, y, :) = [0; 153; 153];
endfor

imshow(map)

disp("Programa calculo trayectoria")