map = imread("mapa.jpg");

dimensions = size(map)

currentPosition = [1, 1];
finalPosition = [dimensions(1), dimensions(2)];

while( sum([currentPosition(1), currentPosition(2)]) < sum([finalPosition(1), finalPosition(2)]))
  
  finalPosition(2) = finalPosition(2) - currentPosition(2);
  
  finalPosition(1) = finalPosition(1) - currentPosition(1);
  
  differenceInX = finalPosition(1);
  differenceInY = finalPosition(2);
  
  if differenceInY < differenceInX
    if(differenceInY < 0)
      currentPosition(2) = currentPosition(2) - 1;  
    else
      currentPosition(2) = currentPosition(2) + 1;
    endif
  elseif differenceInY >= differenceInX
    if(differenceInX < 0)
      currentPosition(1) = currentPosition(1) - 1;  
    else
      currentPosition(1) = currentPosition(1) + 1;
    endif
  endif
  map(currentPosition(1),currentPosition(2), :) =[0; 153; 153];
  
endwhile
%map(i, y, :) = [0; 153; 153];

imshow(map)

disp("Programa calculo trayectoria")