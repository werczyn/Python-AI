# s17093 Piotr Werczyński 16c
import numpy as np
import matplotlib.pyplot as plt


# Wyswietlanie macierzy i rozkładu prawdopodobienstwa
def wyswietl_macierz(macierz):
    print('Macierz: ')
    print('\tP  K  N')
    print('P', macierz[0])
    print('K', macierz[1])
    print('N', macierz[2])
    print('Rozklad prawdopodobienstwa: ')
    print('\t\tP\t\t\tK\t\t\tN')
    print('P', macierz[0] / sum(macierz[0]))
    print('K', macierz[1] / sum(macierz[1]))
    print('N', macierz[2] / sum(macierz[2]))


# funkcja do liczenia wyników
def sprawdz_wynik(wybor_maszyny, wybor_gracza):
    wynik = 0
    if wybor_maszyny == wybor_gracza:
        wynik = 0
    elif wybor_maszyny == 'P':
        if wybor_gracza == 'K':
            wynik += 1
        else:
            wynik -= 1
    elif wybor_maszyny == 'K':
        if wybor_gracza == 'N':
            wynik += 1
        else:
            wynik -= 1
    elif wybor_maszyny == 'N':
        if wybor_gracza == 'P':
            wynik += 1
        else:
            wynik -= 1
    return wynik


def oblicz_prawdopodobienstwo(wybor, macierz):
    if wybor == 'P':
        macierz[2] += 1
    elif wybor == 'K':
        macierz[0] += 1
    elif wybor == 'N':
        macierz[1] += 1
    return macierz


# Funkcja przeprowadzenia gry
def graj(wybor_gracza_poprzedni, wybor_gracza):
    if wybor_gracza_poprzedni == 'P':
        dzielnik = sum(pPrzejscia[0])
        wybor_maszyny = np.random.choice(start, p=pPrzejscia[0] / dzielnik)
        pPrzejscia[0] = oblicz_prawdopodobienstwo(wybor=wybor_gracza, macierz=pPrzejscia[0])
    elif wybor_gracza_poprzedni == 'K':
        dzielnik = sum(pPrzejscia[1])
        wybor_maszyny = np.random.choice(start, p=pPrzejscia[1] / dzielnik)
        pPrzejscia[1] = oblicz_prawdopodobienstwo(wybor=wybor_gracza, macierz=pPrzejscia[1])
    elif wybor_gracza_poprzedni == 'N':
        dzielnik = sum(pPrzejscia[2])
        wybor_maszyny = np.random.choice(start, p=pPrzejscia[2] / dzielnik)
        pPrzejscia[2] = oblicz_prawdopodobienstwo(wybor=wybor_gracza, macierz=pPrzejscia[2])
    wybor_gracza_poprzedni = wybor_gracza
    return wybor_gracza_poprzedni, wybor_gracza, wybor_maszyny


# macierz przejscia
pPrzejscia = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

start = ['P', 'K', 'N']
pStart = [1 / 3, 1 / 3, 1 / 3]
wyborGraczaPoprzedni = np.random.choice(start, p=pStart)

wektorY = []
suma = 0
n = 0
czyTrwa = True

print('GRA Papier, Kamien, Nozyce')
print('wpisz \"HELP\" aby uzyskac pomoc')
# Pogranie inputu od gracza dopoki nie bedzie inputu rownego 'exit'
while czyTrwa:
    wyborGracza = input("Wprowadz wybor: ").upper()
    if wyborGracza == 'EXIT':
        czyTrwa = False
    elif wyborGracza == 'HELP':
        print('Wprowadz: ')
        print('\t\"K\": aby rzucic kamien')
        print('\t\"P\": aby rzucic papier')
        print('\t\"N\": aby rzucic nozyce')
        print('\t\"EXIT\": aby wyjsc')
        print('\t\"HELP\": aby uzyskac pomoc')
    else:
        if wyborGracza != 'P' and wyborGracza != 'K' and wyborGracza != 'N':
            wyborGracza = np.random.choice(start, p=pStart)
        wyborGraczaPoprzedni, wyborGracza, wyborMaszyny = graj(wyborGraczaPoprzedni, wyborGracza)
        suma += sprawdz_wynik(wyborMaszyny, wyborGracza)
        wektorY.append(suma)
        print('Wybor gracza: ' + wyborGracza)
        print('Wybor maszyny: ' + wyborMaszyny)
        n += 1

wyswietl_macierz(pPrzejscia)

if n != 0:
    plt.plot(np.arange(n), wektorY, 'm-')
    plt.show()


# # GRA z komputerem
# n = 20
# suma = 0
# wektorY = []
# print('GRA: z komputerem')
# for i in range(n):
#     wyborGracza = np.random.choice(start, p=pStart)
#     wyborGraczaPoprzedni, wyborGracza, wyborMaszyny = graj(wyborGraczaPoprzedni, wyborGracza)
#     suma += sprawdz_wynik(wyborMaszyny, wyborGracza)
#     wektorY.append(suma)
#     # print('Suma: ' + str(suma))
#     # wyswietl_macierz(pPrzejscia)
#     # print("\n")
#
# wyswietl_macierz(pPrzejscia)
# plt.plot(np.arange(n), wektorY, 'm-')
# plt.show()
