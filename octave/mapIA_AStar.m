map = imread("mapa2.jpg");

dimensions = size(map);

function retvalue = isNodeInList(lista, node)
  sizeOfListWithNodes = size(lista(:,4), 1);
  for i=1:sizeOfListWithNodes
   firstElementOnTheListNotEqualToFirstElementOfNode = lista(:,4){i}(1) != node(:,4){1}(1);
   secondElementOnTheListNotEqualTosecondElementOfNode = lista(:,4){i}(2) != node(:,4){1}(2);
     if(firstElementOnTheListNotEqualToFirstElementOfNode && secondElementOnTheListNotEqualTosecondElementOfNode)
        retvalue = 1;
        break;

     else
        retvalue = 0;
     endif
  endfor
endfunction


function retvalue = getListaVecinos(list, node)
 sizeOfListWithNodes = size(lista(:,4), 1);
  for i=1:sizeOfListWithNodes
    g = nodoActual(i,1){1} + 1;
    h = sqrt(
    retvalue = [[retvalue], {g}];
  endfor
endfunction

posicionInicial = [0, 0];

posicionFinal = [dimensions(1), dimensions(2)];

listaAbierta = [];
listaCerrada = [];
listaVecinos = [];

nodoActual = { 0, 0, 0, posicionInicial};

nodoFinal = { 0, 0, 0, posicionFinal};


listaCerrada = [[nodoActual]; [listaCerrada]];

while isNodeNotInList(listaCerrada, nodoFinal)
    nodoActual = listaCerrada(end, :);
    listaVecinos = getListaVecinos(listaCerrada, nodoActual)
endwhile
