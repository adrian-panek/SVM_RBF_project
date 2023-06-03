# SVM_RBF_project

Struktura folderów i plików:
- dataset: <br />
  Cancer_data.csv - zbiór danych używany do badań

- experiments: <br />
  SVM.py - własnoręcznie zaimplementowany algorytm Support Vector Machine w języku Python w oparciu o bibliotekę Scikit-learn <br />
  evaluation.py - skrypt ewaluacyjny, który pozwala na ewaluację badanych algorytmów algorytmów oraz zebranie wartości metryk i zapisanie ich do pliku <br />
  ttest_cal.py - skrypt, który oblicza statytykę T-Studenta i zapisuje jej wartość do pliku <br />

- results: <br />
  --metrics: <br />
    {nazwa_badanego_algorytmu}_results.npy - wyniki metryk dla badanego algorytmu <br />
  --ttest: <br />
    results{i}.npy - wyniki testu T-Studenta dla konkretnego porównania algorytmu <br />
    
- tests: <br />
  tests.py - testy jednostkowe sprawdzające poprawność działania kodu
  
.gitignore - lista plików, znajdujących się na lokalnym środowisku, które nie powinny znaleźć się w repozytorium
requirements.txt - biblioteki potrzebne do powtórzenia badań we własnym środowisku
