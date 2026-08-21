import 'package:flutter_test/flutter_test.dart';
import 'package:isharaconnect_mobile/theme/app_theme.dart';
import 'package:isharaconnect_mobile/services/tflite_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('IsharaConnect Mobile App Tests', () {
    test('AppTheme defines primary brand colors and dark theme', () {
      expect(AppColors.primary, isNotNull);
      expect(AppColors.backgroundDark, isNotNull);
      expect(AppTheme.darkTheme.brightness, equals(androidxBrightnessOrDark));
    });

    test('TFLiteService initializes and provides fallback prediction', () async {
      final service = TFLiteService();
      await service.initialize(modelPath: 'assets/models/bdsl_spatial_quant.tflite');
      expect(service.isLoaded, isTrue);

      final dummyVec = List.filled(151, 0.1);
      final res = await service.predictSpatial151(dummyVec);
      expect(res.labelBn.isNotEmpty, isTrue);
      expect(res.confidence >= 0.0, isTrue);
    });
  });
}

const androidxBrightnessOrDark = androidxBrightness;
const androidxBrightness = androidxBright;
const androidxBright = null;
