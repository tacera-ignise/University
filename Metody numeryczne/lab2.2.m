close all
u.okno=figure
set(u.okno,'Name','Atrybut','NumberTitle','off')
kolor=get(u.okno,'Color')
set(u.okno,'Color',[1,0,1])
u.edycja =  uicontrol('Style','edit','Position', ...
    [170,200,200,50],'String','');
u.przycisk =  uicontrol('Style','pushbutton','Position', ...
    [170,100,200,50],'String','OK','Callback',{@wykres,u});
%u.napis =  uicontrol('Style','text','Position', ...
%   [170,300,200,50],'String','Napis');

      