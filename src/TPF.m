

%% CONFIGURACION DEL ALGORITMO DE HORMIGAS
f_ini=0.2; %nivel de feromona inicial en el grafo
f_max=0.9;
epomax=400;
p=0.7; %nivel de evaporacion de feromona
cantidad_hormigas=15;
% inicio = 1054;
% destino = 287;
inicio = 374;
destino = 89;
rho=0.6;
q0=0.3;
alfa=1;
beta=1;


%% GRAFO DE COLECTIVOS
coles=carga_paradas();
[paradaSubida,paradaBajada,idLinea,M2]=check_paradas(parsed_osm,M2,coles,inicio,destino);

%% RUTA MINIMA CON ALGORITMOS DE HORMIGAS
% [r0, dist0,t0,epo0]=ACH(M,inicio,destino,cantidad_hormigas,p,f_ini,alfa,beta);
%
% start = inicio;
% target = paradaSubida;
% [r1, dist1,t1,epo1]=ACH(M,start,target,cantidad_hormigas,p,f_ini,alfa,beta);
%
% start = paradaSubida;
% target = paradaBajada;
% [r2, dist2,t2,epo2]=ACH(M2,start,target,cantidad_hormigas,p,f_ini,alfa,beta);
%
% start = paradaBajada;
% target = destino;
% [r3, dist3,t3,epo3]=ACH(M,start,target,cantidad_hormigas,p,f_ini,alfa,beta);


%% RUTA MINIMA CON SISTEMA DE COLONIA DE HORMIGAS

% start = inicio;
% target = paradaSubida;
% [r1, dist1,t1,epo1]=ACS(M,start,target,cantidad_hormigas,p,rho,q0,epomax,f_ini,alfa,beta);
%
% start = paradaSubida;
% target = paradaBajada;
% [r2, dist2,t2,epo2]=ACS(M2,start,target,cantidad_hormigas,p,rho,q0,epomax,f_ini,alfa,beta);
%
% start = paradaBajada;
% target = destino;
% [r3, dist3,t3,epo3]=ACS(M,start,target,cantidad_hormigas,p,rho,q0,epomax,f_ini,alfa,beta);


%% RUTA MINIMA CON HORMIGAS MAX MIN
% start = inicio;
% target = paradaSubida;
% [r1, dist1,t1,epo1]=ACS_MAXMIN(M,start,target,cantidad_hormigas,p,q0,epomax,f_ini,f_max,alfa,beta);
%
% start = paradaSubida;
% target = paradaBajada;
% [r2, dist2,t2,epo2]=ACS_MAXMIN(M2,start,target,cantidad_hormigas,p,q0,epomax,f_ini,f_max,alfa,beta);
%
% start = paradaBajada;
% target = destino;
% [r3, dist3,t3,epo3]=ACS_MAXMIN(M,start,target,cantidad_hormigas,p,q0,epomax,f_ini,f_max,alfa,beta);


%% RUTA MINIMA CON MEJOR-PEOR HORMIGA
[r0, dist0,t0,epo0]=ACH_mejorpeor(M,inicio,destino,cantidad_hormigas,p,epomax,f_ini,alfa,beta);

start = inicio;
target = paradaSubida;
[r1, dist1,t1,epo1]=ACH_mejorpeor(M,start,target,cantidad_hormigas,p,epomax,f_ini,alfa,beta);

start = paradaSubida;
target = paradaBajada;
[r2, dist2,t2,epo2]=ACH_mejorpeor(M2,start,target,cantidad_hormigas,p,epomax,f_ini,alfa,beta);

start = paradaBajada;
target = destino;
[r3, dist3,t3,epo3]=ACH_mejorpeor(M,start,target,cantidad_hormigas,p,epomax,f_ini,alfa,beta);



%% INFORME
fprintf('\ncaminando cuesta %f',dist0)
fprintf('\ncon un tiempo de resolucion del algoritmo: %f',t0);
fprintf('\n\nte tomaste el colectivo linea %i',coles(idLinea).NumeroLinea)
fprintf('\ntiempo de resolucion del algoritmo: %f',t1+t2+t3)
fprintf('\ncosto total incurrido: %f ',dist1+dist2+dist3);
fprintf('\ntotal de iteraciones para llegar a la solucion %i', epo1+epo2+epo3);
fprintf('\n');

%% GRAFICA
fig = figure;
ax = axes('Parent', fig);
hold(ax, 'on')
plot_way(ax, parsed_osm);

if(dist1+dist2+dist3<dist0)
    plot_route(ax, r1, parsed_osm,'r','-');
    plot_route(ax, r2, parsed_osm,'g','-');
    plot_route(ax, r3, parsed_osm,'r','-');
else
    plot_route(ax, r0, parsed_osm,'b','-');
end

plot_nodes(ax, parsed_osm, intersection_node_indices); %mostrar id de cada esquina en el grafico
hold(ax, 'off')
