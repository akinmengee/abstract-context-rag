import 'package:flutter/material.dart';

/// The circuit-brain mark, recolored to any theme color via srcIn blend -
/// same trick as the CSS mask-image on web (icon-white.png's alpha shape,
/// painted with whatever color the surface needs).
class BrandMark extends StatelessWidget {
  const BrandMark({super.key, required this.color, this.size = 40});

  final Color color;
  final double size;

  @override
  Widget build(BuildContext context) {
    return ColorFiltered(
      colorFilter: ColorFilter.mode(color, BlendMode.srcIn),
      child: Image.asset('assets/icon-white.png',
          height: size, fit: BoxFit.contain),
    );
  }
}
