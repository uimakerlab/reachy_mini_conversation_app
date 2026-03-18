import { type Mat4, mat4Identity, mat4Clone, composeWorldOffset } from "./mat4";

/**
 * Compose a primary 4x4 pose with multiple named secondary offsets
 * using proper matrix composition (matching Python's compose_world_offset).
 */
export class PoseComposer {
  private primaryPose: Mat4 = mat4Identity();
  private secondaryOffsets = new Map<string, Mat4>();
  private composedPose: Mat4 = mat4Identity();
  private isDirty = true;

  setPrimary(pose: Mat4): void {
    this.primaryPose = mat4Clone(pose);
    this.isDirty = true;
  }

  setSecondary(name: string, offset: Mat4): void {
    this.secondaryOffsets.set(name, mat4Clone(offset));
    this.isDirty = true;
  }

  removeSecondary(name: string): void {
    if (this.secondaryOffsets.delete(name)) this.isDirty = true;
  }

  compose(): Mat4 {
    if (!this.isDirty) return mat4Clone(this.composedPose);

    // Start with primary pose, then apply each secondary offset
    // using world-frame composition: R_final = R_off @ R_abs, t_final = t_abs + t_off
    let result = mat4Clone(this.primaryPose);
    for (const offset of this.secondaryOffsets.values()) {
      result = composeWorldOffset(result, offset, true);
    }
    this.composedPose = result;
    this.isDirty = false;
    return mat4Clone(result);
  }

  reset(): void {
    this.primaryPose = mat4Identity();
    this.secondaryOffsets.clear();
    this.composedPose = mat4Identity();
    this.isDirty = false;
  }
}
