# SVM_RBF_project

Struktura folderów i plików:
- dataset:
  Cancer_data.csv - zbiór danych używany do badań

- experiments:
  SVM.py - własnoręcznie zaimplementowany algorytm Support Vector Machine w języku Python w oparciu o bibliotekę Scikit-learn
  evaluation.py - skrypt ewaluacyjny, który pozwala na ewaluację badanych algorytmów algorytmów oraz zebranie wartości metryk i zapisanie ich do pliku
  ttest_cal.py - skrypt, który oblicza statytykę T-Studenta i zapisuje jej wartość do pliku

- results:
  --metrics:
    {nazwa_badanego_algorytmu}_results.npy - wyniki metryk dla badanego algorytmu
  --ttest:
    results{i}.npy - wyniki testu T-Studenta dla konkretnego porównania algorytmu
    
- tests:
  tests.py - testy jednostkowe sprawdzające poprawność działania kodu
  
.gitignore - lista plików, znajdujących się na lokalnym środowisku, które nie powinny znaleźć się w repozytorium
requirements.txt - biblioteki potrzebne do powtórzenia badań we własnym środowisku
