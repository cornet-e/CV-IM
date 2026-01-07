import sys
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, 
    QListWidget, QVBoxLayout, QWidget, QLabel, QTableView, QHBoxLayout, QListWidgetItem, QTextEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QColor
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.express as px

# --- Fonctions ---
def calculate_cv(df, value_col='Value', group_col='Lot'):
    results = []
    for grp, data in df.groupby(group_col):
        values = data[value_col].dropna()
        if len(values) == 0: continue
        cv_classic = values.std(ddof=1) / values.mean() * 100
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        cv_robust = (iqr / 1.349) / values.mean() * 100
        results.append({'Lot': grp, 'CV_classic': cv_classic, 'CV_robust': cv_robust})
    return pd.DataFrame(results)

def calculate_bias_eeq(eeq_df):
    results = []
    for param, data in eeq_df.groupby('Paramètre'):
        mean_val = data['Résultat'].mean()
        mean_eeq = data['Moyenne_EEQ'].mean()
        sd_eeq = data['SD_EEQ'].mean()
        bias = mean_val - mean_eeq
        u = np.sqrt((data['Résultat'].std(ddof=1)**2 / len(data)) + (sd_eeq**2))
        U = 2 * u
        results.append({'Paramètre': param, 'Biais': bias, 'U élargie': U})
    return pd.DataFrame(results)

def load_csv(file_path):
    return pd.read_csv(file_path, sep=';', encoding='utf-8')

# --- Classe Dashboard ---
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard CV & Biais - Pro")
        self.setGeometry(30, 30, 1800, 1000)

        self.ciq_df = None
        self.eeq_df = None
        self.cv_df = pd.DataFrame()
        self.bias_df = pd.DataFrame()

        main_layout = QVBoxLayout()

        # --- Boutons chargement ---
        load_layout = QHBoxLayout()
        self.load_ciq_btn = QPushButton("Charger CIQ")
        self.load_ciq_btn.clicked.connect(self.load_ciq)
        self.load_eeq_btn = QPushButton("Charger EEQ")
        self.load_eeq_btn.clicked.connect(self.load_eeq)
        load_layout.addWidget(self.load_ciq_btn)
        load_layout.addWidget(self.load_eeq_btn)
        main_layout.addLayout(load_layout)

        # --- Filtres ---
        filter_layout = QHBoxLayout()
        self.automate_list = QListWidget(); self.automate_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.param_list = QListWidget(); self.param_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.lot_list = QListWidget(); self.lot_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for lst,label in [(self.automate_list,'Automate'),(self.param_list,'Paramètre'),(self.lot_list,'Lot')]:
            v = QVBoxLayout()
            v.addWidget(QLabel(label))
            v.addWidget(lst)
            filter_layout.addLayout(v)
            lst.itemSelectionChanged.connect(self.update_dashboard)
        main_layout.addLayout(filter_layout)

        # --- Tableau résultats ---
        self.table_view = QTableView()
        main_layout.addWidget(QLabel("Tableau des résultats"))
        main_layout.addWidget(self.table_view)

        # --- Graphiques ---
        self.plot_cv_view = QWebEngineView()
        self.plot_bias_view = QWebEngineView()
        main_layout.addWidget(QLabel("Graphique CV par Lot"))
        main_layout.addWidget(self.plot_cv_view)
        main_layout.addWidget(QLabel("Graphique Biais et U élargie"))
        main_layout.addWidget(self.plot_bias_view)

        # --- Résumé stats ---
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        main_layout.addWidget(QLabel("Résumé statistique"))
        main_layout.addWidget(self.summary_text)

        # --- Boutons export ---
        export_layout = QHBoxLayout()
        self.export_btn = QPushButton("Exporter CSV")
        self.export_btn.clicked.connect(self.export_csv)
        self.export_plot_btn = QPushButton("Exporter Graphiques HTML")
        self.export_plot_btn.clicked.connect(self.export_html)
        export_layout.addWidget(self.export_btn)
        export_layout.addWidget(self.export_plot_btn)
        main_layout.addLayout(export_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    # --- Chargement fichiers ---
    def load_ciq(self):
        file_path,_ = QFileDialog.getOpenFileName(self,"Ouvrir fichier CIQ","","CSV Files (*.csv)")
        if file_path:
            self.ciq_df = load_csv(file_path)
            for lst,col in [(self.automate_list,'Automate'),(self.param_list,'Paramètre'),(self.lot_list,'Lot')]:
                lst.clear()
                if col in self.ciq_df.columns:
                    for val in sorted(self.ciq_df[col].dropna().unique()):
                        item = QListWidgetItem(str(val))
                        item.setSelected(True)
                        lst.addItem(item)
            self.update_dashboard()

    def load_eeq(self):
        file_path,_ = QFileDialog.getOpenFileName(self,"Ouvrir fichier EEQ","","CSV Files (*.csv)")
        if file_path:
            self.eeq_df = load_csv(file_path)
            self.update_dashboard()

    # --- Filtrage ---
    def get_selected_values(self,lst):
        return [item.text() for item in lst.selectedItems()]

    # --- Mise à jour dashboard ---
    def update_dashboard(self):
        if self.ciq_df is None: return
        df = self.ciq_df.copy()
        for lst,col in [(self.automate_list,'Automate'),(self.param_list,'Paramètre'),(self.lot_list,'Lot')]:
            selected = self.get_selected_values(lst)
            if selected: df = df[df[col].isin(selected)]
        self.cv_df = calculate_cv(df) if 'Value' in df.columns else pd.DataFrame()
        self.bias_df = calculate_bias_eeq(self.eeq_df) if self.eeq_df is not None else pd.DataFrame()

        # Tableau
        if not self.cv_df.empty and not self.bias_df.empty:
            display_df = pd.merge(self.cv_df,self.bias_df,how='outer',left_on='Lot',right_on='Paramètre')
        elif not self.cv_df.empty:
            display_df = self.cv_df
        else:
            display_df = self.bias_df
        model = QStandardItemModel(display_df.shape[0],display_df.shape[1])
        model.setHorizontalHeaderLabels(display_df.columns.tolist())
        for i in range(display_df.shape[0]):
            for j in range(display_df.shape[1]):
                val = display_df.iat[i,j]
                item = QStandardItem(str(round(val,2)) if isinstance(val,(float,np.float64)) else str(val))
                if display_df.columns[j] in ['CV_classic','CV_robust'] and isinstance(val,(float,np.float64)):
                    item.setBackground(QColor(255,100,100) if val>5 else QColor(100,255,100))
                if display_df.columns[j]=='Biais' and 'U élargie' in display_df.columns:
                    U = display_df.at[i,'U élargie']
                    if isinstance(val,(float,np.float64)) and isinstance(U,(float,np.float64)):
                        if abs(val) > 2*U:
                            item.setBackground(QColor(255,0,0))
                model.setItem(i,j,item)
        self.table_view.setModel(model)

        # Graphiques
        if not self.cv_df.empty:
            fig_cv = px.bar(self.cv_df,x='Lot',y=['CV_classic','CV_robust'],barmode='group',title='CV par Lot')
            self.plot_cv_view.setHtml(fig_cv.to_html(include_plotlyjs='cdn'))
        if not self.bias_df.empty:
            fig_bias = px.bar(self.bias_df,x='Paramètre',y=['Biais','U élargie'],barmode='group',title='Biais et U élargie')
            self.plot_bias_view.setHtml(fig_bias.to_html(include_plotlyjs='cdn'))

        # Résumé statistique
        summary = ""
        if not self.cv_df.empty:
            summary += f"Moyenne CV_classic: {self.cv_df['CV_classic'].mean():.2f}%\n"
            summary += f"CV_classic max: {self.cv_df['CV_classic'].max():.2f}%\n"
            summary += f"Moyenne CV_robust: {self.cv_df['CV_robust'].mean():.2f}%\n"
        if not self.bias_df.empty:
            summary += f"Moyenne Biais: {self.bias_df['Biais'].mean():.2f}\n"
            summary += f"Nombre de paramètres: {self.bias_df.shape[0]}\n"
        self.summary_text.setText(summary)

    # --- Export ---
    def export_csv(self):
        if self.cv_df.empty and self.bias_df.empty: return
        file_path,_ = QFileDialog.getSaveFileName(self,"Exporter CSV","","CSV Files (*.csv)")
        if file_path:
            if not self.cv_df.empty and not self.bias_df.empty:
                export_df = pd.merge(self.cv_df,self.bias_df,how='outer',left_on='Lot',right_on='Paramètre')
            elif not self.cv_df.empty:
                export_df = self.cv_df
            else:
                export_df = self.bias_df
            export_df.to_csv(file_path,sep=';',index=False)
            print(f"Résultats exportés dans {file_path}")

    def export_html(self):
        if not self.cv_df.empty:
            fig_cv = px.bar(self.cv_df,x='Lot',y=['CV_classic','CV_robust'],barmode='group',title='CV par Lot')
            fig_cv.write_html("CV_graph.html")
        if not self.bias_df.empty:
            fig_bias = px.bar(self.bias_df,x='Paramètre',y=['Biais','U élargie'],barmode='group',title='Biais et U élargie')
            fig_bias.write_html("Bias_graph.html")
        print("Graphiques exportés en HTML: CV_graph.html et Bias_graph.html")

# --- Lancer l'application ---
if __name__=="__main__":
    app = QApplication(sys.argv)
    window = Dashboard()
    window.show()
    sys.exit(app.exec())
