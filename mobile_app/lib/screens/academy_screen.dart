import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

class AcademyScreen extends StatefulWidget {
  const AcademyScreen({super.key});

  @override
  State<AcademyScreen> createState() => _AcademyScreenState();
}

class _AcademyScreenState extends State<AcademyScreen> {
  int _xp = 1450;
  int _streak = 7;
  double _selectedLessonAccuracy = 88.5;

  final List<Map<String, dynamic>> _curriculumTiers = [
    {
      'title': 'Tier 1: Alphabets & Digits',
      'bn_title': 'স্তর ১: বর্ণমালা ও সংখ্যা',
      'lessons': 24,
      'completed': 24,
      'color': AppColors.cyanAccent,
      'icon': Icons.sort_by_alpha_rounded,
      'subtopics': ['স্বরবর্ণ (Vowels)', 'ব্যঞ্জনবর্ণ (Consonants)', 'সংখ্যা (Numbers 1-10)'],
    },
    {
      'title': 'Tier 2: Daily Life Vocabulary',
      'bn_title': 'স্তর ২: নিত্যদিনের শব্দাবলী',
      'lessons': 30,
      'completed': 18,
      'color': AppColors.emeraldSuccess,
      'icon': Icons.forum_rounded,
      'subtopics': ['Greetings (শুভেচ্ছা)', 'Family (পরিবার)', 'Emotions & Health (স্বাস্থ্য ও অনুভূতি)'],
    },
    {
      'title': 'Tier 3: BdSL Sentence Grammar',
      'bn_title': 'স্তর ৩: বাক্য গঠন ও ব্যাকরণ',
      'lessons': 16,
      'completed': 6,
      'color': AppColors.indigoAccent,
      'icon': Icons.account_tree_rounded,
      'subtopics': ['Subject-Object-Verb (SOV)', 'Negation & Questions', 'Spatial Reference Frames'],
    },
    {
      'title': 'Tier 4: Master Interpreter Simulation',
      'bn_title': 'স্তর ৪: পেশাদার দোভাষী সিমুলেশন',
      'lessons': 10,
      'completed': 2,
      'color': AppColors.amberWarning,
      'icon': Icons.school_rounded,
      'subtopics': ['News Broadcasting (সংবাদ পরিবেশন)', 'Hospital Emergency (হাসপাতাল জরুরি)', 'Court & Legal (আইনি সহায়তা)'],
    },
  ];

  void _startLessonQuiz(String lessonTitle) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.panelDark,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Row(
          children: [
            const Icon(Icons.psychology_alt_rounded, color: AppColors.cyanAccent),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                'Practice: $lessonTitle',
                style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: AppColors.textPrimary),
              ),
            ),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              height: 120,
              decoration: BoxDecoration(
                color: AppColors.surfaceDark,
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.camera_alt_rounded, color: AppColors.cyanGlow, size: 36),
                    SizedBox(height: 8),
                    Text('DTW Real-Time Motion Tracking', style: TextStyle(fontSize: 12, color: AppColors.textSecondary)),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Target Accuracy:', style: TextStyle(color: AppColors.textSecondary)),
                Text('${_selectedLessonAccuracy.toStringAsFixed(1)}%', style: const TextStyle(color: AppColors.emeraldSuccess, fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 8),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: _selectedLessonAccuracy / 100.0,
                backgroundColor: AppColors.surfaceDark,
                valueColor: const AlwaysStoppedAnimation<Color>(AppColors.emeraldSuccess),
                minHeight: 8,
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel', style: TextStyle(color: AppColors.textMuted)),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(ctx);
              setState(() {
                _xp += 50;
              });
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('⭐ +50 XP Earned! Great Gesture Match.'),
                  backgroundColor: AppColors.emeraldSuccess,
                ),
              );
            },
            child: const Text('Submit Gesture'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bgDark,
      appBar: AppBar(
        title: const Text('BdSL Interpreter Academy'),
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            _buildStatsHeader(),
            const SizedBox(height: 20),
            const Text(
              'CURRICULUM TIERS',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.bold,
                letterSpacing: 1.2,
                color: AppColors.cyanAccent,
              ),
            ),
            const SizedBox(height: 12),
            ..._curriculumTiers.map((tier) => _buildTierCard(tier)),
          ],
        ),
      ),
    );
  }

  Widget _buildStatsHeader() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: AppTheme.glassCardDecoration(borderColor: AppColors.cyanAccent.withOpacity(0.3)),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildStatBadge('XP Score', '$_xp ⭐', AppColors.cyanAccent),
          Container(width: 1, height: 35, color: AppColors.surfaceDark),
          _buildStatBadge('Streak', '$_streak Days 🔥', AppColors.coralError),
          Container(width: 1, height: 35, color: AppColors.surfaceDark),
          _buildStatBadge('Rank', 'Master Tier', AppColors.emeraldSuccess),
        ],
      ),
    );
  }

  Widget _buildStatBadge(String label, String value, Color color) {
    return Column(
      children: [
        Text(value, style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: color)),
        const SizedBox(height: 2),
        Text(label, style: const TextStyle(fontSize: 11, color: AppColors.textSecondary)),
      ],
    );
  }

  Widget _buildTierCard(Map<String, dynamic> tier) {
    final double progress = (tier['completed'] as int) / (tier['lessons'] as int);
    final Color color = tier['color'] as Color;

    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      decoration: AppTheme.glassCardDecoration(),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          leading: Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(color: color.withOpacity(0.2), borderRadius: BorderRadius.circular(10)),
            child: Icon(tier['icon'] as IconData, color: color),
          ),
          title: Text(
            tier['title'] as String,
            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14, color: AppColors.textPrimary),
          ),
          subtitle: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(tier['bn_title'] as String, style: const TextStyle(fontSize: 12, color: AppColors.textSecondary)),
              const SizedBox(height: 6),
              Row(
                children: [
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: progress,
                        backgroundColor: AppColors.surfaceDark,
                        valueColor: AlwaysStoppedAnimation<Color>(color),
                        minHeight: 6,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text('${tier['completed']}/${tier['lessons']}', style: const TextStyle(fontSize: 11, color: AppColors.textMuted)),
                ],
              ),
            ],
          ),
          children: (tier['subtopics'] as List<String>).map((topic) {
            return ListTile(
              dense: true,
              leading: const Icon(Icons.play_circle_outline_rounded, color: AppColors.cyanAccent, size: 20),
              title: Text(topic, style: const TextStyle(fontSize: 13, color: AppColors.textPrimary)),
              trailing: const Icon(Icons.chevron_right_rounded, color: AppColors.textMuted),
              onTap: () => _startLessonQuiz(topic),
            );
          }).toList(),
        ),
      ),
    );
  }
}
