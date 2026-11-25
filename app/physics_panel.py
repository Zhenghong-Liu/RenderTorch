from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PhysicsPanel(QWidget):
    refresh_requested = Signal()

    def __init__(self, controller, refresh=None, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._refresh = refresh
        self.setWindowTitle("物理控制")

        self.type_combo = QComboBox()
        self.type_combo.addItem("刚体", "rigidbody")
        self.type_combo.addItem("布料", "cloth")
        self.type_combo.addItem("软体", "softbody")
        self.algorithm_label = QLabel("布料算法")
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem("质点弹簧", "mass_spring")
        self.algorithm_combo.addItem("PBD", "pbd")
        self.algorithm_combo.addItem("XPBD", "xpbd")
        self.display_combo = QComboBox()
        self.display_combo.addItem("实体", "solid")
        self.display_combo.addItem("线框", "wireframe")
        self.display_combo.addItem("粒子", "particles")

        self.start_button = QPushButton("开始")
        self.pause_button = QPushButton("暂停")
        self.single_button = QPushButton("单步")
        self.reset_button = QPushButton("重置")
        buttons = QHBoxLayout()
        for button in (
            self.start_button,
            self.pause_button,
            self.single_button,
            self.reset_button,
        ):
            buttons.addWidget(button)

        self.gravity_input = QDoubleSpinBox()
        self.gravity_input.setRange(-30.0, 10.0)
        self.gravity_input.setSingleStep(0.1)
        self.gravity_input.setSuffix(" m/s²")
        self.primary_label = QLabel("结构刚度")
        self.primary_input = QDoubleSpinBox()
        self.primary_input.setDecimals(6)
        self.primary_input.setSingleStep(0.01)
        self.iterations_input = QSpinBox()
        self.iterations_input.setRange(1, 64)
        self.iterations_label = QLabel("约束迭代")
        self.metrics_label = QLabel()
        self.metrics_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("模拟类型", self.type_combo)
        form.addRow(self.algorithm_label, self.algorithm_combo)
        form.addRow("显示方式", self.display_combo)
        form.addRow("重力", self.gravity_input)
        form.addRow(self.primary_label, self.primary_input)
        form.addRow(self.iterations_label, self.iterations_input)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("性能与稳定性"))
        layout.addWidget(self.metrics_label)
        layout.addStretch()

        self.type_combo.currentIndexChanged.connect(self._change_type)
        self.algorithm_combo.currentIndexChanged.connect(self._change_algorithm)
        self.display_combo.currentIndexChanged.connect(self._change_display)
        self.start_button.clicked.connect(self._start)
        self.pause_button.clicked.connect(self._pause)
        self.single_button.clicked.connect(self._single_step)
        self.reset_button.clicked.connect(self._reset)
        self.gravity_input.valueChanged.connect(self._change_gravity)
        self.primary_input.editingFinished.connect(self._change_primary)
        self.iterations_input.valueChanged.connect(self._change_iterations)
        self.refresh_requested.connect(self._external_refresh)
        self._sync_controls()
        self.update_metrics()

    def _change_type(self) -> None:
        self.controller.set_simulation_type(self.type_combo.currentData())
        self._sync_controls()
        self._request_refresh()

    def _change_algorithm(self) -> None:
        self.controller.set_cloth_algorithm(self.algorithm_combo.currentData())
        self._sync_controls()
        self._request_refresh()

    def _change_display(self) -> None:
        self.controller.set_display_mode(self.display_combo.currentData())
        self._request_refresh()

    def _start(self) -> None:
        self.controller.start()
        self._request_refresh()

    def _pause(self) -> None:
        self.controller.pause()
        self._request_refresh()

    def _single_step(self) -> None:
        self.controller.single_step()
        self._request_refresh()

    def _reset(self) -> None:
        self.controller.reset()
        self._request_refresh()

    def _change_gravity(self, value: float) -> None:
        self.controller.set_gravity(value)
        self.update_metrics()

    def _change_primary(self) -> None:
        self.controller.set_primary_parameter(self.primary_input.value())
        self._request_refresh()

    def _change_iterations(self, value: int) -> None:
        self.controller.set_iterations(value)
        self._request_refresh()

    def _sync_controls(self) -> None:
        physics = self.controller.physics
        for widget, value in (
            (self.type_combo, physics.simulation_type),
            (self.algorithm_combo, physics.cloth_algorithm),
            (self.display_combo, physics.display_mode),
            (self.gravity_input, physics.gravity_value),
            (self.primary_input, physics.primary_parameter),
            (self.iterations_input, physics.iterations),
        ):
            widget.blockSignals(True)
            if isinstance(widget, QComboBox):
                widget.setCurrentIndex(widget.findData(value))
            else:
                widget.setValue(value)
            widget.blockSignals(False)

        is_cloth = physics.simulation_type == "cloth"
        self.algorithm_label.setVisible(is_cloth)
        self.algorithm_combo.setVisible(is_cloth)
        uses_iterations = is_cloth or physics.simulation_type == "softbody"
        self.iterations_label.setVisible(uses_iterations)
        self.iterations_input.setVisible(uses_iterations)

        if physics.simulation_type == "rigidbody":
            label, maximum, decimals, step = "恢复系数", 1.0, 2, 0.05
        elif physics.simulation_type == "softbody":
            label, maximum, decimals, step = "距离柔度", 1.0, 7, 0.000001
        elif physics.cloth_algorithm == "mass_spring":
            label, maximum, decimals, step = "结构刚度", 5000.0, 2, 10.0
        elif physics.cloth_algorithm == "pbd":
            label, maximum, decimals, step = "PBD 刚度", 1.0, 3, 0.01
        else:
            label, maximum, decimals, step = "结构柔度", 1.0, 7, 0.000001
        self.primary_label.setText(label)
        self.primary_input.setRange(0.0, maximum)
        self.primary_input.setDecimals(decimals)
        self.primary_input.setSingleStep(step)
        self.primary_input.setValue(physics.primary_parameter)

    def update_metrics(self) -> None:
        metrics = self.controller.metrics()
        if not metrics:
            return
        lines = [
            f"步数：{metrics['steps']}",
            f"模拟时间：{metrics['time']:.3f} s",
            f"单步耗时：{metrics['step_ms']:.2f} ms",
            f"最大速度：{metrics['max_speed']:.3f} m/s",
            f"约束误差：{metrics['constraint_error']:.4f}",
            f"接触粒子：{metrics['contacts']}",
        ]
        if "volume_error" in metrics:
            lines.append(f"体积误差：{metrics['volume_error']:.4f}")
        self.metrics_label.setText("\n".join(lines))

    def _request_refresh(self) -> None:
        self.update_metrics()
        if self._refresh is not None:
            self._refresh()
        self.refresh_requested.emit()

    def _external_refresh(self) -> None:
        self.update_metrics()
