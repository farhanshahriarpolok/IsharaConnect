import 'dart:async';
import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../services/tflite_service.dart';
import '../services/websocket_service.dart';

class SignerScreen extends StatefulWidget {
  const SignerScreen({super.key});

  @override
  State<SignerScreen> createState() => _SignerScreenState();
}

class _SignerScreenState extends State<SignerScreen> with SingleTickerProviderStateMixin {
  final TFLiteService _tfliteService = TFLiteService();
  late WebSocketService _wsService;

  String _currentGlossBn = 'ধন্যবাদ';
  String _currentGlossEn = 'Thank you';
  double _confidence = 0.94;
  bool _isStable = true;
  String _roomId = 'room_public_01';
  ConnectionStatus _connectionStatus = ConnectionStatus.connected;

  final TextEditingController _textController = TextEditingController();
  final List<String> _sentenceHistory = ['স্বাগতম', 'কেমন আছেন'];
  Timer? _simTimer;

  late AnimationController _pulseController;

  @override
  void initState() {
    super.initState();
    _tfliteService.initialize();
    _wsService = WebSocketService(roomId: _roomId, clientType: 'signer');
    _wsService.connect();

    _wsService.statusStream.listen((status) {
      if (mounted) {
        setState(() => _connectionStatus = status);
      }
    });

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1),
    )..repeat(reverse: true);

    // Realistic landmark inference simulation loop
    _simTimer = Timer.periodic(const Duration(milliseconds: 1500), (timer) {
      if (mounted) {
        final mockSigns = [
          {'bn': 'ধন্যবাদ', 'en': 'Thank you', 'conf': 0.96},
          {'bn': 'কেমন আছেন', 'en': 'How are you', 'conf': 0.91},
          {'bn': 'সাহায্য', 'en': 'Help', 'conf': 0.88},
          {'bn': 'আমি ভালো আছি', 'en': 'I am fine', 'conf': 0.94},
        ];
        final sample = mockSigns[timer.tick % mockSigns.length];
        setState(() {
          _currentGlossBn = sample['bn'] as String;
          _currentGlossEn = sample['en'] as String;
          _confidence = sample['conf'] as double;
          _isStable = true;
          _sentenceHistory.add(_currentGlossBn);
          if (_sentenceHistory.length > 5) {
            _sentenceHistory.removeAt(0);
          }
        });
        _wsService.sendSignEvent({
          'label_bn': _currentGlossBn,
          'label_en': _currentGlossEn,
          'confidence': _confidence,
          'is_stable': _isStable,
        });
      }
    });
  }

  @override
  void dispose() {
    _simTimer?.cancel();
    _pulseController.dispose();
    _tfliteService.dispose();
    _wsService.dispose();
    _textController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bgDark,
      body: SafeArea(
        child: Column(
          children: [
            _buildTopBar(),
            Expanded(
              child: Stack(
                children: [
                  _buildCameraViewfinder(),
                  _buildLandmarkOverlay(),
                  _buildSubtitleTickerHUD(),
                ],
              ),
            ),
            _buildSentenceHistoryBar(),
            _buildQuickInputBar(),
          ],
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      color: AppColors.panelDark,
      child: Row(
        children: [
          Row(
            children: [
              FadeTransition(
                opacity: _pulseController,
                child: Container(
                  width: 10,
                  height: 10,
                  decoration: BoxDecoration(
                    color: _connectionStatus == ConnectionStatus.connected
                        ? AppColors.emeraldSuccess
                        : AppColors.coralError,
                    shape: BoxShape.circle,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Text(
                _connectionStatus == ConnectionStatus.connected ? 'ONLINE' : 'RECONNECTING',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: _connectionStatus == ConnectionStatus.connected
                      ? AppColors.emeraldSuccess
                      : AppColors.coralError,
                ),
              ),
            ],
          ),
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: AppColors.surfaceDark,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              'Room: $_roomId',
              style: const TextStyle(fontSize: 12, color: AppColors.textSecondary),
            ),
          ),
          const SizedBox(width: 12),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: AppColors.surfaceDark,
              borderRadius: BorderRadius.circular(6),
            ),
            child: const Text(
              '32ms • 30 FPS',
              style: TextStyle(fontSize: 11, fontFamily: 'monospace', color: AppColors.cyanAccent),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCameraViewfinder() {
    return Container(
      width: double.infinity,
      height: double.infinity,
      color: Colors.black87,
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.videocam_rounded, size: 48, color: AppColors.textMuted.withOpacity(0.5)),
            const SizedBox(height: 12),
            Text(
              'Live Camera Stream • Dual Hand Active',
              style: TextStyle(color: AppColors.textMuted.withOpacity(0.7), fontSize: 13),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLandmarkOverlay() {
    return CustomPaint(
      size: Size.infinite,
      painter: _LandmarkVisualizerPainter(),
    );
  }

  Widget _buildSubtitleTickerHUD() {
    return Positioned(
      bottom: 20,
      left: 16,
      right: 16,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: AppTheme.glassCardDecoration(
          borderColor: AppColors.cyanAccent.withOpacity(0.4),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'LIVE TRANSLATION HUD',
                  style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: AppColors.cyanAccent, letterSpacing: 1.2),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: AppColors.emeraldSuccess.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    '${(_confidence * 100).toStringAsFixed(1)}% Match',
                    style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: AppColors.emeraldSuccess),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              _currentGlossBn,
              style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: AppColors.textPrimary),
            ),
            Text(
              _currentGlossEn,
              style: const TextStyle(fontSize: 14, color: AppColors.textSecondary, fontStyle: FontStyle.italic),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSentenceHistoryBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      color: AppColors.panelDark,
      child: Row(
        children: [
          const Icon(Icons.history_rounded, size: 18, color: AppColors.textMuted),
          const SizedBox(width: 8),
          Expanded(
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: _sentenceHistory.map((phrase) {
                  return Container(
                    margin: const EdgeInsets.only(right: 8),
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: AppColors.surfaceDark,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(phrase, style: const TextStyle(fontSize: 12, color: AppColors.textSecondary)),
                  );
                }).toList(),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQuickInputBar() {
    return Container(
      padding: const EdgeInsets.all(12),
      color: AppColors.bgDark,
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _textController,
              decoration: const InputDecoration(
                hintText: 'Type message to synthesize gestures...',
                contentPadding: EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              ),
            ),
          ),
          const SizedBox(width: 8),
          IconButton.filled(
            onPressed: () {
              if (_textController.text.isNotEmpty) {
                _wsService.sendSpeechEvent(_textController.text);
                _textController.clear();
              }
            },
            icon: const Icon(Icons.send_rounded),
            style: IconButton.styleFrom(backgroundColor: AppColors.cyanAccent, foregroundColor: AppColors.bgDark),
          ),
        ],
      ),
    );
  }
}

class _LandmarkVisualizerPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paintDot = Paint()..color = AppColors.cyanGlow..style = PaintingStyle.fill;
    final paintLine = Paint()..color = AppColors.cyanAccent.withOpacity(0.5)..strokeWidth = 2;

    // Draw simulated 151D dual-hand skeleton lines
    final centerL = Offset(size.width * 0.35, size.height * 0.45);
    final centerR = Offset(size.width * 0.65, size.height * 0.45);

    for (int i = 0; i < 5; i++) {
      final tipL = Offset(centerL.dx + (i - 2) * 16, centerL.dy - 35);
      final tipR = Offset(centerR.dx + (i - 2) * 16, centerR.dy - 35);

      canvas.drawLine(centerL, tipL, paintLine);
      canvas.drawLine(centerR, tipR, paintLine);

      canvas.drawCircle(tipL, 4, paintDot);
      canvas.drawCircle(tipR, 4, paintDot);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
