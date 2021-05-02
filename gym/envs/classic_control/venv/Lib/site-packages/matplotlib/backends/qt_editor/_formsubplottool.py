from matplotlib.backends.qt_compat import QtWidgets


class UiSubplotTool(QtWidgets.QDialog):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName("SubplotTool")
        self._widgets = {}

        main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(main_layout)

        for group, spinboxes, buttons in [
                ("Borders",
                 ["top", "bottom", "left", "right"], ["Export values"]),
                ("Spacings",
                 ["hspace", "wspace"], ["Tight layout", "Reset", "Close"]),
        ]:
            layout = QtWidgets.QVBoxLayout()
            main_layout.addLayout(layout)
            box = QtWidgets.QGroupBox(group)
            layout.addWidget(box)
            inner = QtWidgets.QFormLayout(box)
            for name in spinboxes:
                self._widgets[name] = widget = QtWidgets.QDoubleSpinBox()
                widget.setMinimum(0)
                widget.setMaximum(1)
                widget.setDecimals(3)
                widget.setSingleStep(0.005)
                widget.setKeyboardTracking(False)
                inner.addRow(name, widget)
            layout.addStretch(1)
            for name in buttons:
                self._widgets[name] = widget = QtWidgets.QPushButton(name)
                # Don't trigger on <enter>, which is used to input values.
                widget.setAutoDefault(False)
                layout.addWidget(widget)

        self._widgets["Close"].setFocus()
