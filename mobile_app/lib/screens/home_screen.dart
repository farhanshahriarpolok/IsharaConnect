import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import 'signer_screen.dart';
import 'speaker_screen.dart';
import 'academy_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0;

  final List<Widget> _screens = [
    const SignerScreen(),
    const SpeakerScreen(),
    const AcademyScreen(),
    const _ScenarioSimulatorView(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: AppColors.panelDark,
          border: const Border(top: BorderSide(color: AppColors.glassBorder, width: 1)),
          boxShadow: [
            BoxShadow(color: Colors.black.withOpacity(0.3), blurRadius: 12, offset: const Offset(0, -2)),
          ],
        ),
        child: NavigationBar(
          selectedIndex: _currentIndex,
          onDestinationSelected: (idx) => setState(() => _currentIndex = idx),
          backgroundColor: Colors.transparent,
          indicatorColor: AppColors.cyanAccent.withOpacity(0.2),
          elevation: 0,
          destinations: const [
            NavigationDestination(
              icon: Icon(Icons.front_hand_rounded, color: AppColors.textSecondary),
              selectedIcon: Icon(Icons.front_hand_rounded, color: AppColors.cyanAccent),
              label: 'Signer',
            ),
            NavigationDestination(
              icon: Icon(Icons.record_voice_over_rounded, color: AppColors.textSecondary),
              selectedIcon: Icon(Icons.record_voice_over_rounded, color: AppColors.cyanAccent),
              label: 'Speaker',
            ),
            NavigationDestination(
              icon: Icon(Icons.school_rounded, color: AppColors.textSecondary),
              selectedIcon: Icon(Icons.school_rounded, color: AppColors.cyanAccent),
              label: 'Academy',
            ),
            NavigationDestination(
              icon: Icon(Icons.emergency_rounded, color: AppColors.textSecondary),
              selectedIcon: Icon(Icons.emergency_rounded, color: AppColors.coralError),
              label: 'Simulator',
            ),
          ],
        ),
      ),
    );
  }
}

class _ScenarioSimulatorView extends StatelessWidget {
  const _ScenarioSimulatorView();

  final List<Map<String, dynamic>> _scenarios = const [
    {
      'title': 'Hospital & Clinic Emergency',
      'bn': 'হাসপাতাল জরুরি বিভাগ',
      'icon': Icons.local_hospital_rounded,
      'color': AppColors.coralError,
      'desc': 'Communicate acute symptoms, pain levels, and allergy history.'
    },
    {
      'title': 'Police & Safety Station',
      'bn': 'থানা ও আইনি নিরাপত্তা',
      'icon': Icons.local_police_rounded,
      'color': AppColors.indigoAccent,
      'desc': 'Report incidents, theft, or request immediate emergency protection.'
    },
    {
      'title': 'Fire & Disaster Relief',
      'bn': 'দমকল ও দুর্যোগকালীন সহায়তা',
      'icon': Icons.local_fire_department_rounded,
      'color': AppColors.amberWarning,
      'desc': 'SOS coordinates transmission and rapid relief requests.'
    },
    {
      'title': 'Bank & Public Services',
      'bn': 'ব্যাংক ও নাগরিক সেবা',
      'icon': Icons.account_balance_rounded,
      'color': AppColors.emeraldSuccess,
      'desc': 'Account verification, KYC authentication, and service inquiries.'
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bgDark,
      appBar: AppBar(
        title: const Text('Scenario Simulator (জরুরি সিমুলেটর)'),
      ),
      body: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: _scenarios.length,
        itemBuilder: (context, idx) {
          final item = _scenarios[idx];
          final Color color = item['color'] as Color;

          return Container(
            margin: const EdgeInsets.only(bottom: 16),
            padding: const EdgeInsets.all(16),
            decoration: AppTheme.glassCardDecoration(),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(item['icon'] as IconData, color: color, size: 28),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        item['title'] as String,
                        style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14, color: AppColors.textPrimary),
                      ),
                      Text(
                        item['bn'] as String,
                        style: const TextStyle(fontSize: 12, color: AppColors.cyanAccent),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        item['desc'] as String,
                        style: const TextStyle(fontSize: 11, color: AppColors.textSecondary),
                      ),
                    ],
                  ),
                ),
                const Icon(Icons.arrow_forward_ios_rounded, size: 14, color: AppColors.textMuted),
              ],
            ),
          );
        },
      ),
    );
  }
}
