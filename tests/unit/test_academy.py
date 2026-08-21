import pytest
from PyQt6.QtWidgets import QApplication
import sys

from desktop_app.ui.academy_dashboard import AcademyDashboard
from desktop_app.ui.scenario_simulator import ScenarioSimulator

# Basic application instance for tests
@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

def test_academy_dashboard_initialization(qapp):
    dashboard = AcademyDashboard()
    assert dashboard is not None
    assert dashboard.tree.topLevelItemCount() == 4
    
def test_scenario_simulator_initialization(qapp):
    simulator = ScenarioSimulator()
    assert simulator is not None
    assert simulator.scenario_list.count() == 4
