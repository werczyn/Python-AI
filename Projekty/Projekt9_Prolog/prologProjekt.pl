%definicja poczatkowego warunku
:- dynamic gdzie/1.
gdzie(kuchnia).

%pokoje
pokoj(sypialnia).
pokoj(kuchnia).
pokoj(lazienka).

%swiatlo
:- dynamic swiatlo/2.
swiatlo(sypialnia,wylaczone).
swiatlo(kuchnia,wlaczone).
swiatlo(lazienka,wylaczone).

%drzwi
drzwi(kuchnia,sypialnia).
drzwi(sypialnia,kuchnia).

drzwi(sypialnia,lazienka).
drzwi(lazienka,sypialnia).

%przedmioty
:- dynamic przedmiot/2.
przedmiot(biurko,sypialnia).
przedmiot(krzeslo,sypialnia).
przedmiot(lozko,sypialnia).
przedmiot(telefon,sypialnia).

przedmiot(kuchenka,kuchnia).
przedmiot(piekarnik,kuchnia).
przedmiot(talerz,kuchnia).

przedmiot(sedes,lazienka).
przedmiot(prysznic,lazienka).

lekki(krzeslo).
lekki(talerz).
lekki(telefon).


%Operacje
jest_przejscie(Miejsce):-
    gdzie(X),
    pokoj(Miejsce),
    drzwi(X,Miejsce).

przejdz(Miejsce):-
    jest_przejscie(Miejsce),
    retract(gdzie(X)),
    asserta(gdzie(Miejsce)),
    write('Przeszedlem do '),
    write(Miejsce), nl.

sprawdzPrzejscia:-
    gdzie(X),
    drzwi(X,Miejsce),
    format('Z ~w jest przejscie do ~w',[X,Miejsce]).

sprawdzGdzie:-
    gdzie(Miejsce),
    write("Jestes w: "), write(Miejsce),nl.

sprawdzPrzedmioty:-
    gdzie(Miejsce),
    swiatlo(Miejsce,wlaczone),
    przedmiot(Przedmiot,Miejsce),
    format("W ~w jest ~w ~n",[Miejsce,Przedmiot]).

sprawdzPrzedmioty:-
    gdzie(Miejsce),
    swiatlo(Miejsce,wylaczone),
    write('Wlacz swiatlo aby zobaczyc').


sprawdzLekkie:-
    gdzie(Miejsce),
    swiatlo(Miejsce,wlaczone),
    lekki(Przedmiot),
    przedmiot(Przedmiot,Miejsce),
    format('Mozesz przeniesc ~w',[Przedmiot]).

sprawdzLekkie:-
    gdzie(Miejsce),
    swiatlo(Miejsce,wylaczone),
    write('Wlacz swiatlo aby zobaczyc').

przeniesPrzedmiot(Przedmiot,Miejsce):-
    lekki(Przedmiot),
    gdzie(X),
    przedmiot(Przedmiot,X),
    retract(przedmiot(Przedmiot,X)),
    asserta(przedmiot(Przedmiot,Miejsce)),
    przejdz(Miejsce),
    format('Przeniesiono przedmiot ~w, z ~w do ~w',[Przedmiot,X,Miejsce]).

sprawdzSwiatlo:-
    gdzie(Miejsce),
    swiatlo(Miejsce,Stan),
    format('swiatlo ~w w ~w',[Stan,Miejsce]).

zapalSwiatlo:-
    gdzie(Miejsce),
    swiatlo(Miejsce,wylaczone),
    retract(swiatlo(Miejsce,wylaczone)),
    asserta(swiatlo(Miejsce,wlaczone)),
    format('Zapalono swiatlo w ~w',[Miejsce]).

zgasSwiatlo:-
    gdzie(Miejsce),
    swiatlo(Miejsce,wlaczone),
    retract(swiatlo(Miejsce,wlaczone)),
    asserta(swiatlo(Miejsce,wylaczone)),
    format('Zgaszono swiatlo w ~w',[Miejsce]).

wyswietlPierwszy:-
    gdzie(Miejsce),
    swiatlo(Miejsce,wlaczone),
    przedmiot(Przedmiot,Miejsce),
    format("W ~w jest ~w ~n",[Miejsce,Przedmiot]),!.

