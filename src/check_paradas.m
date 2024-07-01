function [nodo_start, nodo_end, NumLinea,X] = check_paradas(parsed_osm,X,Lineas,inicio,destino)
    
    opciones=struct;
    for i=1:size(Lineas,2)
        %obtener latitudes y longitudes de paradas, punto inicio y final
        lat_lon=get_unique_node_xy(parsed_osm, Lineas(i).recorrido);
        LL_inicio=get_unique_node_xy(parsed_osm, inicio);
        LL_destino=get_unique_node_xy(parsed_osm, destino);
        
        %distancia(1,:) origen->paradaSubida. distancia(2,:) paradaBajada->destino
        distancias=zeros(2,size(lat_lon.id,2));
        for j=1:size(lat_lon.id,2)
            distancias(1,j)=haversine(lat_lon.id(2,j),lat_lon.id(1,j),LL_inicio.id(2),LL_inicio.id(1));% lat1,lon1,lat2,lon2
            distancias(2,j)=haversine(lat_lon.id(2,j),lat_lon.id(1,j),LL_destino.id(2),LL_destino.id(1));
        end
        
        [~,pos1]=min(distancias(1,:));
        [~,pos2]=min(distancias(2,:));
        opciones(i).id=i;
        opciones(i).nodos=[Lineas(i).recorrido(pos1) Lineas(i).recorrido(pos2)];
        opciones(i).dists=[distancias(1,pos1) distancias(2,pos2)]; %distancia [origen-parada1, parada2-destino]
    end
    
    %seleccionar las 2 mejores lineas de colectivos
    distancias=zeros(2,size(Lineas,2));
    for i=1:size(distancias,2)
        distancias(:,i)=opciones(i).dists;
    end
    [~,pos1]=min( distancias(1,:) );
    [~,pos2]=min( distancias(2,:) );
    
    if(pos1==pos2) %el mismo cole con el que me subo, me deja cerca del destino
        nodo_start=opciones(pos1).nodos(1);
        nodo_end=opciones(pos1).nodos(2);
        NumLinea = opciones(pos1).id;
    else %subo a un cole, pero el otro me deja mas cerca. Entonces elijo la linea que me minimice la distancia
        suma1=sum(opciones(pos1).dists);
        suma2=sum(opciones(pos2).dists);
        if(suma1<=suma2)
            nodo_start=opciones(pos1).nodos(1);
            nodo_end=opciones(pos1).nodos(2);
            NumLinea = opciones(pos1).id;
        else
            nodo_start=opciones(pos2).nodos(1);
            nodo_end=opciones(pos2).nodos(2);
            NumLinea = opciones(pos2).id;
        end
    end
    
    %configurar grafo de colectivos
    peso=0.5;
    costo_subirCole=3;
    costo_esperarCole=2;
    for i=1:length(Lineas(NumLinea).recorrido)-1
        if(Lineas(NumLinea).recorrido(i)==nodo_start)
            X(Lineas(NumLinea).recorrido(i),Lineas(NumLinea).recorrido(i+1))=peso+costo_subirCole + costo_esperarCole;
            X(Lineas(NumLinea).recorrido(i+1),Lineas(NumLinea).recorrido(i))=peso+costo_subirCole + costo_esperarCole;
        else
            X(Lineas(NumLinea).recorrido(i),Lineas(NumLinea).recorrido(i+1))=peso;
            X(Lineas(NumLinea).recorrido(i+1),Lineas(NumLinea).recorrido(i))=peso;
        end
    end
end