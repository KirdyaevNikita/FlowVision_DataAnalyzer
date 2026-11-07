"""Графический анализ пользовательских величин FlowVision.

Зависимости:
    pip install pandas matplotlib PySide6

"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PySide6.QtGui import QIcon
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal, QItemSelection
from PySide6.QtGui import QFont, QColor, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QLineEdit,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QGroupBox,
    QSizePolicy,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QDoubleSpinBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure

ICON_PATH = Path(__file__).resolve().parent / "icon.svg"

class DataFrameModel(QAbstractTableModel):
    """Лёгкая модель таблицы: не создаёт отдельный Qt-объект для каждой ячейки."""

    def __init__(self, dataframe: pd.DataFrame | None = None):
        super().__init__()
        self._dataframe = dataframe if dataframe is not None else pd.DataFrame()

    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        self.beginResetModel()
        self._dataframe = dataframe
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._dataframe)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._dataframe.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        value = self._dataframe.iat[index.row(), index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            if pd.isna(value):
                return ""
            if isinstance(value, (float,)):
                return f"{value:.7g}"
            return str(value)

        if role == Qt.ItemDataRole.ToolTipRole:
            return str(value)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._dataframe.columns[section])
            return str(section + 1)
        if role == Qt.ItemDataRole.ToolTipRole and orientation == Qt.Orientation.Horizontal:
            return str(self._dataframe.columns[section])
        return None


class SelectableHeader(QHeaderView):
    """Заголовок с выбором колонок обычным и Ctrl-щелчком."""

    column_clicked = Signal(int, object)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            section = self.logicalIndexAt(event.position().toPoint())
            if section >= 0:
                self.column_clicked.emit(section, event.modifiers())
                event.accept()
                return
        super().mousePressEvent(event)


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(7, 4), dpi=100, tight_layout=True, facecolor="white")
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()


class FlowVisionAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.setWindowTitle("FlowVision Data Analyzer")
        self.resize(1500, 950)

        self.data: pd.DataFrame | None = None
        self.file_path: Path | None = None
        self.selected_columns: list[int] = []  # порядок: X, затем Y
        self.iterations: pd.Series | None = None
        self.interval_summary: list[dict] = []
        self.summary_columns: list[str] = []

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(9)

        # Верхняя панель
        toolbar = QHBoxLayout()
        self.load_button = QPushButton("🗁  Загрузить файл")
        self.load_button.setMinimumHeight(38)
        self.load_button.clicked.connect(self.load_file)
        toolbar.addWidget(self.load_button)

        self.file_label = QLabel("Файл не загружен")
        self.file_label.setObjectName("muted")
        toolbar.addWidget(self.file_label, 1)

        self.info_label = QLabel("Ожидание файла")
        self.info_label.setObjectName("muted")
        toolbar.addWidget(self.info_label)
        root.addLayout(toolbar)

        # Таблица предпросмотра
        table_group = QGroupBox("Предварительный просмотр данных")
        table_layout = QVBoxLayout(table_group)
        self.selection_label = QLabel(
            "Выберите X щелчком по заголовку, затем выберите Y с зажатым Ctrl."
        )
        self.selection_label.setObjectName("muted")
        table_layout.addWidget(self.selection_label)

        self.table = QTableView()
        self.table_model = DataFrameModel()
        self.table.setModel(self.table_model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectColumns)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setDefaultSectionSize(23)
        self.table.verticalHeader().setFixedWidth(55)
        self.header = SelectableHeader(Qt.Orientation.Horizontal, self.table)
        self.header.setSectionsClickable(True)
        self.header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.header.setDefaultSectionSize(230)
        self.header.setMinimumSectionSize(90)
        self.header.setFixedHeight(58)
        self.table.setHorizontalHeader(self.header)
        self.header.column_clicked.connect(self.handle_column_click)
        table_layout.addWidget(self.table)
        root.addWidget(table_group, 5)

        # Параметры расчёта
        controls = QGroupBox("Параметры осреднения")
        controls_layout = QHBoxLayout(controls)
        controls_layout.addWidget(QLabel("Длина интервала, итераций:"))
        self.averaging_spin = QSpinBox()
        self.averaging_spin.setRange(1, 10_000_000)
        self.averaging_spin.setValue(500)
        self.averaging_spin.setMinimumWidth(120)
        self.averaging_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        controls_layout.addWidget(self.averaging_spin)

        controls_layout.addSpacing(18)
        controls_layout.addWidget(QLabel("Границы интервалов (через запятую):"))
        self.boundaries_edit = QLineEdit("2000, 4000, 6000, 8000")
        self.boundaries_edit.setPlaceholderText("Например: 2000, 4000, 6000")
        controls_layout.addWidget(self.boundaries_edit, 1)

        self.create_table_button = QPushButton("Создать таблицу")
        self.create_table_button.setMinimumHeight(36)
        self.create_table_button.setEnabled(False)
        self.create_table_button.clicked.connect(self.create_summary_table)
        controls_layout.addWidget(self.create_table_button)

        self.plot_button = QPushButton("Построить графики")
        self.plot_button.setMinimumHeight(36)
        self.plot_button.setEnabled(False)
        self.plot_button.clicked.connect(self.build_plots)
        controls_layout.addWidget(self.plot_button)
        root.addWidget(controls)

        # Два графика
        plots_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.convergence_canvas = PlotCanvas()
        self.history_canvas = PlotCanvas()
        plots_splitter.addWidget(self._canvas_group("Отклонения средних величин", self.convergence_canvas))
        plots_splitter.addWidget(self._canvas_group("История изменения величины", self.history_canvas))
        plots_splitter.setStretchFactor(0, 1)
        plots_splitter.setStretchFactor(1, 1)
        root.addWidget(plots_splitter, 5)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Готово к загрузке файла FlowVision")

    @staticmethod
    def _canvas_group(title: str, canvas: PlotCanvas) -> QWidget:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 8, 6, 6)
        toolbar = NavigationToolbar2QT(canvas, group)
        toolbar.setObjectName("plotToolbar")
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        return group

    def _apply_style(self):
        # Тёмная оболочка в духе VS Code; область самих графиков остаётся белой.
        self.setStyleSheet("""
            QWidget { color: #d4d4d4; }
            QMainWindow, QWidget#centralWidget { background: #252526; }
            QGroupBox {
                background: #2d2d30; color: #d4d4d4;
                border: 1px solid #454545; border-radius: 6px;
                margin-top: 9px; font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px;
                color: #d4d4d4;
            }
            QPushButton {
                background: #0e639c; color: #ffffff; border: 1px solid #1177bb;
                border-radius: 4px; padding: 7px 14px; font-weight: 600;
            }
            QPushButton:hover { background: #1177bb; }
            QPushButton:disabled { background: #3a3d41; color: #858585; border-color: #454545; }
            QLineEdit, QSpinBox {
                background: #3c3c3c; color: #f0f0f0;
                border: 1px solid #555555; border-radius: 4px; padding: 6px;
                selection-background-color: #264f78;
            }
            QTableView {
                background: #1e1e1e; alternate-background-color: #252526;
                color: #d4d4d4; gridline-color: #3f3f46;
                selection-background-color: #264f78; selection-color: #ffffff;
            }
            QTableView::item { padding: 2px; }
            QHeaderView::section {
                background: #333337; color: #d4d4d4;
                border: 1px solid #454545; padding: 5px; font-weight: 600;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #252526; border: none; margin: 0;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: #555555; border-radius: 4px; min-width: 18px; min-height: 18px;
            }
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: #707070; }
            QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
            QSplitter::handle { background: #252526; }
            QSplitter::handle:horizontal { width: 1px; }
            QLabel#muted { color: #9da0a5; }
            QStatusBar { background: #252526; color: #bdbdbd; }
            QToolBar#plotToolbar {
                background: #333337; border: 1px solid #454545;
                spacing: 4px; padding: 2px;
            }
            QToolBar#plotToolbar QToolButton {
                background: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555555; border-radius: 3px;
                padding: 3px;
            }
            QToolBar#plotToolbar QToolButton:hover { background: #505050; }
            QToolBar#plotToolbar QToolButton:checked { background: #0e639c; }
        """)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл пользовательских величин FlowVision",
            str(self.file_path.parent if self.file_path else Path.home()),
            "Текстовые файлы (*.txt);;Все файлы (*.*)",
        )
        if not path:
            return

        try:
            dataframe = pd.read_csv(
                path,
                sep="\t",
                encoding="utf-8-sig",
                decimal=".",
                low_memory=False,
            )
            if dataframe.empty or len(dataframe.columns) < 2:
                raise ValueError("В файле недостаточно данных или колонок.")
            if len(dataframe.columns) < 2:
                raise ValueError("Для анализа нужны как минимум две колонки.")

            iteration_values = pd.to_numeric(dataframe.iloc[:, 0], errors="coerce")
            if iteration_values.notna().sum() == 0:
                raise ValueError("Не удалось прочитать номера шагов из первой колонки.")

        except Exception as error:
            QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить файл:\n{error}")
            return

        self.data = dataframe
        self.file_path = Path(path)
        self.iterations = iteration_values
        self.table_model.set_dataframe(dataframe)
        self.table.clearSelection()
        self.selected_columns = [0]
        self._apply_column_selection()

        self.file_label.setText(self.file_path.name)
        self.info_label.setText(f"{len(dataframe):,} строк  ·  {len(dataframe.columns)} колонок".replace(",", " "))
        self.plot_button.setEnabled(False)
        self.create_table_button.setEnabled(True)
        self.interval_summary = []
        self._clear_canvases()
        self.statusBar().showMessage("Файл загружен. Выберите две колонки для построения графиков.")
        self.table.setColumnWidth(0, 125)
        self.table.horizontalScrollBar().setValue(0)

    def handle_column_click(self, column: int, modifiers):
        if self.data is None:
            return

        ctrl_pressed = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        if not ctrl_pressed:
            self.selected_columns = [column]
        elif column in self.selected_columns:
            self.selected_columns.remove(column)
        elif len(self.selected_columns) == 0:
            self.selected_columns = [column]
        elif len(self.selected_columns) == 1:
            self.selected_columns.append(column)
        else:
            # X остаётся первым выбранным столбцом, новый Ctrl-щелчок заменяет Y.
            self.selected_columns[1] = column

        self._apply_column_selection()
        self.plot_button.setEnabled(len(self.selected_columns) == 2)

    def _apply_column_selection(self):
        selection_model = self.table.selectionModel()
        selection_model.clearSelection()
        if self.data is None:
            return

        for position, column in enumerate(self.selected_columns):
            if self.table_model.rowCount() == 0:
                continue
            selection = QItemSelection(
                self.table_model.index(0, column),
                self.table_model.index(self.table_model.rowCount() - 1, column),
            )
            flag = QItemSelectionModelFlag.select_columns()
            selection_model.select(selection, flag)

        names = [str(self.data.columns[i]) for i in self.selected_columns]
        if len(names) == 0:
            message = "Колонки не выбраны. Щёлкните по заголовку X."
        elif len(names) == 1:
            message = f"X: {names[0]}   |   Теперь выберите Y с зажатым Ctrl."
        else:
            message = f"X: {names[0]}   |   Y: {names[1]}"
        self.selection_label.setText(message)

    def _clear_canvases(self):
        for canvas, title in (
            (self.convergence_canvas, "Отклонения средних величин"),
            (self.history_canvas, "История изменения величины"),
        ):
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)
            ax.set_title(title)
            ax.text(0.5, 0.5, "Выберите колонки и постройте графики", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_axis_off()
            canvas.draw_idle()

    def _read_boundaries(self) -> list[int]:
        raw = self.boundaries_edit.text().strip()
        if not raw:
            return []
        try:
            values = [int(part.strip()) for part in raw.replace(";", ",").split(",") if part.strip()]
        except ValueError as error:
            raise ValueError("Границы должны быть целыми числами, разделёнными запятыми.") from error
        if any(value <= 0 for value in values):
            raise ValueError("Границы интервалов должны быть положительными.")
        return sorted(set(values))

    def build_plots(self):
        if self.data is None or len(self.selected_columns) != 2:
            QMessageBox.warning(self, "Не выбраны колонки", "Выберите X и Y в заголовках таблицы.")
            return

        x_column = self.data.columns[self.selected_columns[0]]
        y_column = self.data.columns[self.selected_columns[1]]
        iteration_column = self.data.columns[0]

        try:
            averaging_length = self.averaging_spin.value()
            boundaries = self._read_boundaries()
            iteration_values = pd.to_numeric(self.data[iteration_column], errors="coerce")
            valid_iterations = iteration_values.dropna()
            if valid_iterations.empty:
                raise ValueError("В первой колонке нет корректных номеров шагов.")
            last_iteration = int(valid_iterations.max())

            if last_iteration not in boundaries:
                boundaries.append(last_iteration)
            boundaries = sorted(set(boundaries))
            intervals = [
                (end - averaging_length + 1, end)
                for end in boundaries
                if end <= last_iteration
            ]
            if not intervals:
                raise ValueError("Нет интервалов для анализа. Проверьте заданные границы.")

            x_values = pd.to_numeric(self.data[x_column], errors="coerce")
            y_values = pd.to_numeric(self.data[y_column], errors="coerce")
            numeric = pd.DataFrame({
                "iteration": iteration_values,
                "x": x_values,
                "y": y_values,
            })
            numeric = numeric.dropna(subset=["iteration", "x", "y"])

            averages = []
            interval_data_list = []
            for number, (start, end) in enumerate(intervals, start=1):
                part = numeric[(numeric["iteration"] >= start) & (numeric["iteration"] <= end)]
                if part.empty:
                    continue
                averages.append({
                    "stage": number,
                    "start": start,
                    "end": end,
                    "count": len(part),
                    "mean_x": part["x"].mean(),
                    "mean_y": part["y"].mean(),
                })
                interval_data_list.append((start, end, part))

            if not averages:
                raise ValueError("В заданных интервалах нет строк с числовыми значениями X и Y.")

            for index, item in enumerate(averages[:-1]):
                next_y = averages[index + 1]["mean_y"]
                item["difference"] = ((item["mean_y"] - next_y) / abs(next_y)) * 100 if next_y != 0 else None
            averages[-1]["difference"] = None

            self._plot_convergence(averages, str(x_column), str(y_column))
            self._plot_history(numeric, interval_data_list, str(x_column), str(y_column))
            self.statusBar().showMessage(
                f"Графики построены. Использовано интервалов: {len(averages)}; последняя итерация: {last_iteration}."
            )

        except Exception as error:
            QMessageBox.warning(self, "Не удалось построить графики", str(error))

    def _get_intervals(self) -> tuple[pd.Series, list[tuple[int, int]], int]:
        if self.data is None:
            raise ValueError("Сначала загрузите файл FlowVision.")
        iteration_values = pd.to_numeric(self.data.iloc[:, 0], errors="coerce")
        valid_iterations = iteration_values.dropna()
        if valid_iterations.empty:
            raise ValueError("В первой колонке нет корректных номеров шагов.")
        last_iteration = int(valid_iterations.max())
        averaging_length = self.averaging_spin.value()
        boundaries = self._read_boundaries()
        if last_iteration not in boundaries:
            boundaries.append(last_iteration)
        boundaries = sorted(set(value for value in boundaries if value <= last_iteration))
        intervals = [(end - averaging_length + 1, end) for end in boundaries]
        if not intervals:
            raise ValueError("Нет интервалов для анализа. Проверьте заданные границы.")
        return iteration_values, intervals, last_iteration

    def create_summary_table(self):
        """Формирует средние и относительные изменения по всем физическим колонкам."""
        if self.data is None:
            return
        try:
            iteration_values, intervals, last_iteration = self._get_intervals()
            # Первые две колонки — номер шага и время; остальные — пользовательские величины.
            source_columns = list(self.data.columns[2:])
            if not source_columns:
                raise ValueError("В файле нет пользовательских величин после колонок номера шага и времени.")

            numeric_columns = {
                column: pd.to_numeric(self.data[column], errors="coerce")
                for column in source_columns
            }
            rows = []
            for number, (start, end) in enumerate(intervals, start=1):
                mask = iteration_values.between(start, end, inclusive="both")
                interval_iterations = iteration_values[mask].dropna()
                if interval_iterations.empty:
                    continue
                means = {}
                for column, values in numeric_columns.items():
                    selected = values[mask].dropna()
                    means[column] = float(selected.mean()) if not selected.empty else None
                rows.append({
                    "number": number,
                    "left": start,
                    "mean_iteration": float(interval_iterations.mean()),
                    "right": end,
                    "means": means,
                })

            if not rows:
                raise ValueError("В заданных интервалах нет данных для осреднения.")

            # Отклонение текущего среднего от среднего следующего интервала, %.
            for index, row in enumerate(rows):
                deviations = {}
                next_means = rows[index + 1]["means"] if index + 1 < len(rows) else None
                for column in source_columns:
                    current = row["means"].get(column)
                    next_value = next_means.get(column) if next_means is not None else None
                    if current is None or next_value is None or next_value == 0:
                        deviations[column] = None
                    else:
                        deviations[column] = abs((current - next_value) / abs(next_value)) * 100
                row["deviations"] = deviations

            self.interval_summary = rows
            self.summary_columns = [str(column) for column in source_columns]
            self._show_summary_dialog(source_columns, rows)
            self.statusBar().showMessage(
                f"Сводная таблица сформирована: {len(rows)} интервалов, {len(source_columns)} величин."
            )
        except Exception as error:
            QMessageBox.warning(self, "Не удалось создать таблицу", str(error))

    def _show_summary_dialog(self, source_columns, rows):
        dialog = QDialog(self)
        dialog.setWindowIcon(QIcon(str(ICON_PATH)))
        dialog.setWindowTitle("Сводная таблица осреднения FlowVision")
        dialog.resize(1450, 800)
        dialog.setMinimumSize(900, 500)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        fixed_headers = [
            "№ интервала",
            "Граница интервала (левая)",
            "Среднее значение интервала",
            "Граница интервала (правая)",
        ]
        column_count = len(fixed_headers) + 2 * len(source_columns)
        table = QTableWidget(len(rows) + 2, column_count, dialog)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setVisible(False)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

        for col, header in enumerate(fixed_headers):
            item = QTableWidgetItem(header)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setToolTip(header)
            table.setItem(0, col, item)
            table.setSpan(0, col, 2, 1)
            table.setColumnWidth(col, [90, 175, 190, 175][col])

        for index, column in enumerate(source_columns):
            col = 4 + index * 2
            group_item = QTableWidgetItem(str(column))
            group_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            group_item.setToolTip(str(column))
            table.setItem(0, col, group_item)
            table.setSpan(0, col, 1, 2)
            mean_header = QTableWidgetItem("Среднее значение")
            dev_header = QTableWidgetItem("Отклонение, %")
            mean_header.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            dev_header.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(1, col, mean_header)
            table.setItem(1, col + 1, dev_header)
            table.setColumnWidth(col, 135)
            table.setColumnWidth(col + 1, 115)

        for row_index, record in enumerate(rows, start=2):
            fixed_values = [
                record["number"], record["left"], record["mean_iteration"], record["right"]
            ]
            for col, value in enumerate(fixed_values):
                item = QTableWidgetItem(self._format_table_value(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row_index, col, item)
            for index, column in enumerate(source_columns):
                col = 4 + index * 2
                mean = record["means"].get(column)
                deviation = record["deviations"].get(column)
                mean_item = QTableWidgetItem(self._format_table_value(mean))
                deviation_item = QTableWidgetItem(self._format_table_value(deviation))
                mean_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                deviation_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if deviation is not None:
                    deviation_item.setToolTip("Отклонение от среднего значения следующего интервала")
                table.setItem(row_index, col, mean_item)
                table.setItem(row_index, col + 1, deviation_item)

        layout.addWidget(table, 1)
        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("Порог отклонения, %:"))
        threshold = QDoubleSpinBox()
        threshold.setRange(0, 1_000_000)
        threshold.setDecimals(3)
        threshold.setValue(2.0)
        threshold.setSuffix(" %")
        threshold.setFixedWidth(125)
        bottom.addWidget(threshold)

        highlight_button = QPushButton("Выделить")
        highlight_button.clicked.connect(
            lambda: self._highlight_deviations(table, rows, source_columns, threshold.value())
        )
        bottom.addWidget(highlight_button)
        bottom.addStretch(1)

        download_button = QPushButton("Скачать XLSX")
        download_button.clicked.connect(lambda: self._export_summary_xlsx(source_columns, rows, dialog))
        bottom.addWidget(download_button)
        layout.addLayout(bottom)

        dialog.setStyleSheet(self.styleSheet() + """
            QDialog { background: #252526; }
            QTableWidget {
                background: #1e1e1e; alternate-background-color: #252526;
                color: #d4d4d4; gridline-color: #454545;
                selection-background-color: #264f78; selection-color: white;
            }
            QTableWidget::item { padding: 5px; }
        """)
        # Заголовки визуально выделены; все ячейки остаются редактированию недоступны.
        for r in (0, 1):
            for c in range(column_count):
                item = table.item(r, c)
                if item is not None:
                    item.setBackground(QBrush(QColor("#333337")))
                    item.setForeground(QBrush(QColor("#ffffff")))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
        dialog.exec()

    @staticmethod
    def _format_table_value(value):
        if value is None or pd.isna(value):
            return ""
        if isinstance(value, (int,)):
            return str(value)
        return f"{float(value):.8g}"

    def _highlight_deviations(self, table, rows, source_columns, threshold):
        green = QColor("#245b38")
        red = QColor("#7a3030")
        default = QColor("#1e1e1e")
        alternate = QColor("#252526")
        for row_index, record in enumerate(rows, start=2):
            for index, column in enumerate(source_columns):
                item = table.item(row_index, 5 + index * 2)
                value = record["deviations"].get(column)
                if item is None:
                    continue
                if value is None:
                    item.setBackground(QBrush(alternate if row_index % 2 else default))
                elif value <= threshold:
                    item.setBackground(QBrush(green))
                else:
                    item.setBackground(QBrush(red))

    def _export_summary_xlsx(self, source_columns, rows, dialog):
        path, _ = QFileDialog.getSaveFileName(
            dialog,
            "Сохранить сводную таблицу",
            str((self.file_path.parent if self.file_path else Path.home()) / "FlowVision_summary.xlsx"),
            "Книга Excel (*.xlsx)",
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils import get_column_letter

            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Осреднение"
            fixed_headers = [
                "№ интервала", "Граница интервала (левая)",
                "Среднее значение интервала", "Граница интервала (правая)"
            ]
            for col, header in enumerate(fixed_headers, start=1):
                sheet.merge_cells(start_row=1, start_column=col, end_row=2, end_column=col)
                sheet.cell(1, col, header)
            for index, column in enumerate(source_columns):
                col = 5 + index * 2
                sheet.merge_cells(start_row=1, start_column=col, end_row=1, end_column=col + 1)
                sheet.cell(1, col, str(column))
                sheet.cell(2, col, "Среднее значение")
                sheet.cell(2, col + 1, "Отклонение, %")

            for row_index, record in enumerate(rows, start=3):
                fixed_values = [record["number"], record["left"], record["mean_iteration"], record["right"]]
                for col, value in enumerate(fixed_values, start=1):
                    sheet.cell(row_index, col, value)
                for index, column in enumerate(source_columns):
                    col = 5 + index * 2
                    sheet.cell(row_index, col, record["means"].get(column))
                    sheet.cell(row_index, col + 1, record["deviations"].get(column))

            header_fill = PatternFill("solid", fgColor="333337")
            header_font = Font(color="FFFFFF", bold=True)
            thin_gray = Side(style="thin", color="777777")
            for row in sheet.iter_rows(min_row=1, max_row=2 + len(rows)):
                for cell in row:
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                    cell.border = Border(left=thin_gray, right=thin_gray, top=thin_gray, bottom=thin_gray)
                    if cell.row <= 2:
                        cell.fill = header_fill
                        cell.font = header_font
            for col, width in enumerate([13, 23, 25, 23], start=1):
                sheet.column_dimensions[get_column_letter(col)].width = width
            for index in range(len(source_columns)):
                sheet.column_dimensions[get_column_letter(5 + index * 2)].width = 20
                sheet.column_dimensions[get_column_letter(6 + index * 2)].width = 16
            sheet.freeze_panes = "E3"
            sheet.auto_filter.ref = f"A2:{get_column_letter(4 + 2 * len(source_columns))}{2 + len(rows)}"
            workbook.save(path)
            QMessageBox.information(dialog, "Экспорт завершён", f"Таблица сохранена:\n{path}")
        except Exception as error:
            QMessageBox.critical(dialog, "Ошибка экспорта", f"Не удалось сохранить XLSX:\n{error}")

    @staticmethod
    def _difference_text_and_color(difference):
        """Возвращает подписанное отклонение и цвет по его знаку."""
        if difference is None or pd.isna(difference):
            return None, None
        value = float(difference)
        color = "#d62728" if value > 0 else "#2471c8" if value < 0 else "#222222"
        return f"{value:+.4f} %", color

    def _plot_convergence(self, averages, x_label: str, y_label: str):
        canvas = self.convergence_canvas
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111, facecolor="white")
        canvas.figure.set_facecolor("white")

        xs = [item["mean_x"] for item in averages]
        ys = [item["mean_y"] for item in averages]

        # Тонкая чёрная линия и квадратные маркеры средних значений.
        ax.plot(
            xs, ys, color="black", marker="s", markersize=5,
            markerfacecolor="black", markeredgecolor="black",
            linewidth=0.8, zorder=3,
        )

        for item in averages:
            x_value = item["mean_x"]
            y_value = item["mean_y"]

            # Среднее значение — чёрным непосредственно над маркером.
            ax.annotate(
                f"{y_value:.6g}", (x_value, y_value),
                xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, color="black",
            )

            # Подписанное отклонение — отдельной строкой выше среднего.
            diff_label, diff_color = self._difference_text_and_color(item.get("difference"))
            if diff_label is not None:
                ax.annotate(
                    diff_label, (x_value, y_value),
                    xytext=(0, 21), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=diff_color,
                )

        ax.set_title("Отклонения средних значений", fontsize=10, color="black")
        ax.set_xlabel(f"{x_label}", fontsize=8, color="black", labelpad=3)
        ax.set_ylabel(f"{y_label}", fontsize=8, color="black", labelpad=3)
        ax.tick_params(axis="both", labelsize=8, colors="black")
        ax.grid(True, linestyle="--", alpha=0.45)
        canvas.figure.tight_layout()
        canvas.draw_idle()

    def _plot_history(self, numeric: pd.DataFrame, interval_data_list, x_label: str, y_label: str):
        canvas = self.history_canvas
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111, facecolor="white")
        canvas.figure.set_facecolor("white")

        ax.plot(
            numeric["x"], numeric["y"], linewidth=0.85,
            label=f"Y({x_label})", zorder=1, color="black",
        )
        colors = ["#a8d69a", "#f2c477", "#9dbbea", "#e9a6c1", "#c1a8e8", "#8fd1d1"]

        for index, (start, end, part) in enumerate(interval_data_list):
            color = colors[index % len(colors)]
            mean_y = float(part["y"].mean())
            xmin, xmax = float(part["x"].min()), float(part["x"].max())

            interval_label = f"Интервал {index + 1}"
            if xmin < xmax:
                ax.axvspan(
                    xmin, xmax, color=color, alpha=0.34, zorder=0,
                    label=interval_label,
                )
            else:
                ax.axvline(
                    xmin, color=color, alpha=0.72, linewidth=5,
                    label=interval_label,
                )

            ax.scatter(part["x"], part["y"], s=7, color=color, edgecolors="none", zorder=2)
            if xmin == xmax:
                ax.axhline(mean_y, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
                x_center = xmin
            else:
                ax.hlines(
                    mean_y, xmin, xmax, colors="black",
                    linestyles="dashed", linewidth=1.1, zorder=3,
                )
                x_center = (xmin + xmax) / 2.0

            # Среднее значение над выделенным интервалом.
            ax.annotate(
                f"{mean_y:.6g}", (x_center, mean_y),
                xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, color="black",
                zorder=5,
            )

            # Отклонение текущего среднего от следующего интервала.
            if index + 1 < len(interval_data_list):
                next_part = interval_data_list[index + 1][2]
                next_mean = float(next_part["y"].mean())
                difference = (
                    (mean_y - next_mean) / abs(next_mean) * 100.0
                    if next_mean != 0 else None
                )
                diff_label, diff_color = self._difference_text_and_color(difference)
                if diff_label is not None:
                    ax.annotate(
                        diff_label, (x_center, mean_y),
                        xytext=(0, 21), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=diff_color,
                        zorder=5,
                    )

        ax.set_title("История изменения величины", fontsize=10, color="black")
        ax.set_xlabel(x_label, fontsize=8, color="black", labelpad=3)
        ax.set_ylabel(y_label, fontsize=8, color="black", labelpad=3)
        ax.tick_params(axis="both", labelsize=8, colors="black")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(
            fontsize=7, title_fontsize=7, ncols=2,
        )
        canvas.figure.tight_layout()
        canvas.draw_idle()


class QItemSelectionModelFlag:
    """Небольшой адаптер, чтобы явно задавать выделение целых колонок."""
    @staticmethod
    def select_columns():
        from PySide6.QtCore import QItemSelectionModel
        return QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Columns


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FlowVision Data Analyzer")
    app.setFont(QFont("Segoe UI", 9))
    window = FlowVisionAnalyzer()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
