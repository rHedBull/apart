/**
 * StepNavigation - Prev/Next controls for navigating simulation steps
 */

import Button from '@cloudscape-design/components/button';
import SpaceBetween from '@cloudscape-design/components/space-between';
import Box from '@cloudscape-design/components/box';

interface StepNavigationProps {
  currentStep: number | null;
  maxSteps: number | null;
  onStepChange: (step: number | null) => void;
}

export function StepNavigation({ currentStep, maxSteps, onStepChange }: StepNavigationProps) {
  if (!maxSteps) return null;

  const canGoPrev = currentStep !== null && currentStep > 1;
  const canGoNext = currentStep !== null && currentStep < maxSteps;
  const isFiltered = currentStep !== null;

  const handlePrev = () => {
    if (currentStep !== null && currentStep > 1) {
      onStepChange(currentStep - 1);
    }
  };

  const handleNext = () => {
    if (currentStep !== null && currentStep < maxSteps) {
      onStepChange(currentStep + 1);
    } else if (currentStep === null) {
      onStepChange(1);
    }
  };

  const handleClear = () => {
    onStepChange(null);
  };

  return (
    <SpaceBetween direction="horizontal" size="xs" alignItems="center">
      <Button
        iconName="angle-left"
        variant="icon"
        disabled={!canGoPrev}
        onClick={handlePrev}
        ariaLabel="Previous step"
      />
      <Box variant="span" fontSize="body-s">
        {isFiltered ? `Step ${currentStep} of ${maxSteps}` : `All ${maxSteps} steps`}
      </Box>
      <Button
        iconName="angle-right"
        variant="icon"
        disabled={!canGoNext}
        onClick={handleNext}
        ariaLabel="Next step"
      />
      {isFiltered && (
        <Button variant="inline-link" onClick={handleClear}>
          Show all
        </Button>
      )}
    </SpaceBetween>
  );
}
