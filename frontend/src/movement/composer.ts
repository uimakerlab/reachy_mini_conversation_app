import { type HeadPose, createNeutralPose, addPoses, clonePose } from "./types";

export class PoseComposer {
  private primaryPose: HeadPose = createNeutralPose();
  private secondaryOffsets = new Map<string, HeadPose>();
  private composedPose: HeadPose = createNeutralPose();
  private isDirty = true;

  setPrimary(pose: HeadPose): void {
    this.primaryPose = clonePose(pose);
    this.isDirty = true;
  }

  setSecondary(name: string, offset: HeadPose): void {
    this.secondaryOffsets.set(name, clonePose(offset));
    this.isDirty = true;
  }

  removeSecondary(name: string): void {
    if (this.secondaryOffsets.delete(name)) this.isDirty = true;
  }

  compose(): HeadPose {
    if (!this.isDirty) return clonePose(this.composedPose);
    let result = clonePose(this.primaryPose);
    for (const offset of this.secondaryOffsets.values()) {
      result = addPoses(result, offset);
    }
    this.composedPose = result;
    this.isDirty = false;
    return clonePose(result);
  }

  reset(): void {
    this.primaryPose = createNeutralPose();
    this.secondaryOffsets.clear();
    this.composedPose = createNeutralPose();
    this.isDirty = false;
  }
}
