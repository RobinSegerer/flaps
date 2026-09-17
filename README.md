# FLAPs

**Forschungs- und Lern-App für angewandte Psychometrie**

FLAPs ist eine interaktive Streamlit-App zum explorativen Lernen zentraler Konzepte der Statistik und Psychometrie.

## Inhalte

- Mittelwert und Varianz
- Kovarianz, Korrelation und Standardisierung
- Lineare Regression
- Summenscores
- Klassische Testtheorie
- Itemanalyse

## Lokal starten

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deployment auf Streamlit Community Cloud

1. Dieses Repository in GitHub anlegen.
2. In Streamlit Community Cloud das Repository auswählen.
3. `app.py` als Main file path verwenden.
4. Deploy starten.

Das Logo `flaps_logo.png` liegt im selben Ordner wie `app.py`.
