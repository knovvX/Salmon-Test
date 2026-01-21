# Usability Test Protocol
## Fish Origin Classification System

---

## Test Overview

**Objective**: Evaluate the usability and interpretability of the Fish Origin Classification web application for identifying salmon origin (Hatchery vs. Natural) using scale images and biological metadata.

**Duration**: 30-45 minutes per participant

**Test Type**: Moderated usability testing with think-aloud protocol

---

## Participant Profile

### Target Users
- Marine biologists
- Fisheries researchers
- Wildlife resource managers
- Laboratory technicians with experience in salmon identification

### Recruitment Criteria
- Experience with salmon research or classification (preferred but not required)
- Basic computer literacy
- No prior exposure to this specific system

### Sample Size
- 5-8 participants (recommended for identifying major usability issues)

---

## Test Setup

### Materials Needed
1. **Test device**: Computer with modern web browser (Chrome, Firefox, Safari)
2. **Test data package**: 
   - ZIP file containing:
     - 12 salmon scale TIFF images (6_2X.tiff, 14_2X.tiff, 42_2X.tiff, etc.)
     - CSV metadata file (selected_samples_info.csv)
   - Provided to participants at start of test
3. **Recording equipment**: Screen recording software, audio recorder
4. **Consent forms**: Participant consent and data privacy agreements

### System Requirements
- **Backend**: Flask API running on http://127.0.0.1:5001
- **Frontend**: Streamlit app running on http://localhost:8501
- **Model**: Pre-loaded best.ckpt checkpoint

---

## Pre-Test Briefing (5 minutes)

### Introduction Script
*"Thank you for participating in this usability study. Today you'll be testing a web-based system designed to classify salmon as either Hatchery-origin or Natural-origin based on scale images and biological measurements.*

*This system uses machine learning to analyze scale patterns and provide classification predictions with confidence scores. Your feedback will help us improve the interface and make it more useful for researchers like yourself.*

*Please remember:*
- *There are no wrong answers - we're testing the system, not you*
- *Think aloud as you work - tell us what you're thinking, what confuses you, what you expect to happen*
- *Feel free to ask questions, but I may not answer immediately as we want to see how you naturally interact with the system*
- *The session will be recorded for analysis purposes only*"

### Background Questions (Optional)
1. How familiar are you with salmon biology and classification? (1-5 scale)
2. Have you used machine learning or AI tools in your research before?
3. What tools do you currently use for salmon identification?

---

## Task Scenarios

### Task 1: Data Upload (5-7 minutes)

**Scenario**: 
*"You have received a collection of salmon scale images and their associated metadata (fork length, sex, etc.). Your goal is to determine which fish are hatchery-origin and which are natural-origin. Use the system to upload and process this data."*

**Instructions to Participant**:
1. Navigate to the application homepage
2. Upload the provided ZIP file (salmon_test_data.zip)
3. Review the data preview
4. Proceed to the next step when ready

**Observer Notes**:
- [ ] Did participant find the upload interface immediately?
- [ ] Did participant understand what file format was expected?
- [ ] Were there any confusion points during upload?
- [ ] Did participant notice the data quality indicators (matched images, missing values)?
- [ ] Time to complete: _____

**Follow-up Questions**:
- "What did you expect to happen after uploading the file?"
- "Was the information displayed after upload helpful? What would you add or remove?"
- "Did you notice any issues with your data? How did the system communicate this?"

---

### Task 2: Initiate Inference and Review Results (8-10 minutes)

**Scenario**: 
*"Now that your data is loaded, run the classification analysis and examine the results."*

**Instructions to Participant**:
1. Start the inference process
2. Navigate to the results view
3. Examine the overall results
4. Use the filtering/sorting options if needed

**Observer Notes**:
- [ ] Did participant understand how to start inference?
- [ ] Were progress indicators clear during processing?
- [ ] Did participant understand the results table layout?
- [ ] Did participant explore filtering/sorting options?
- [ ] Time to complete: _____

**Follow-up Questions**:
- "What do you think the system is showing you in this results table?"
- "How confident do you feel in these predictions?"
- "Is there any information missing that you would want to see?"

---

### Task 3: Interpret Predictions (10-12 minutes)

**Scenario**: 
*"Select a few samples and examine their detailed predictions. Try to understand what the system is telling you about each fish."*

**Instructions to Participant**:
1. Click on 2-3 different samples to view details
2. Examine the prediction, confidence scores, and Grad-CAM visualizations
3. Compare samples with different confidence levels

**Observer Notes**:
- [ ] Did participant understand how to access detailed views?
- [ ] Did participant interpret the confidence scores correctly?
- [ ] Did participant notice/understand the Grad-CAM heatmaps?
- [ ] Did participant compare multiple samples?
- [ ] Time to complete: _____

**Follow-up Questions**:
- "In your own words, what is the system telling you about this fish?"
- "What does the confidence score mean to you?"
- "What do you think the colored heatmap (Grad-CAM) is showing?"
- "How would you explain this result to a colleague?"

**Probe Questions**:
- Select a high-confidence prediction (>90%):
  - "How would you use this result? Would you trust it?"
  - "Would you need to verify this manually?"
  
- Select a moderate-confidence prediction (60-80%):
  - "What would you do with this result?"
  - "Would this be useful, or would you discard it?"
  
- Select a low-confidence prediction (<60%):
  - "What does this tell you?"
  - "How should borderline cases be handled?"

---

### Task 4: Decision-Making Based on Outputs (8-10 minutes)

**Scenario**: 
*"Imagine you need to make decisions about your research based on these results. Walk me through how you would use this information."*

**Discussion Prompts**:

1. **Classification Decisions**:
   - "If you had 100 samples like these, how would you use the confidence scores to decide which predictions to trust?"
   - "Would you set a confidence threshold? If so, what would it be and why?"
   - "What would you do with samples below your threshold?"

2. **Grad-CAM Interpretation**:
   - "The heatmap shows which parts of the scale image the model focused on. How does this influence your trust in the prediction?"
   - "If the heatmap highlights unexpected areas (e.g., edges, background), would that change your interpretation?"

3. **Metadata Integration**:
   - "The system uses fork length (FL) and sex along with the image. How important is it to see this information in the results?"
   - "Should the system explain how these factors influenced the prediction?"

4. **Workflow Integration**:
   - "How would you integrate this tool into your current workflow?"
   - "What would you do before using this tool? What would you do after?"
   - "Would you use this as a first-pass screening tool, a confirmation tool, or something else?"

5. **Error Handling**:
   - "What would you do if you disagreed with a high-confidence prediction?"
   - "If you had to manually verify a subset of results, which ones would you choose?"

**Observer Notes**:
- [ ] Participant's confidence threshold preference: _____
- [ ] Participant's trust level in predictions (1-5): _____
- [ ] Key decision-making factors mentioned: _____
- [ ] Suggested workflow improvements: _____

---

## Post-Test Questionnaire

### System Usability Scale (SUS)
Rate the following statements on a scale of 1 (Strongly Disagree) to 5 (Strongly Agree):

1. I think that I would like to use this system frequently
2. I found the system unnecessarily complex
3. I thought the system was easy to use
4. I think I would need support from a technical person to use this system
5. I found the various functions in this system were well integrated
6. I thought there was too much inconsistency in this system
7. I imagine most people would learn to use this system very quickly
8. I found the system very cumbersome to use
9. I felt very confident using the system
10. I needed to learn a lot of things before I could get going with this system

### Trust and Interpretability (Custom Questions)

Rate on a scale of 1 (Not at all) to 5 (Extremely):

11. How well do you understand how the system makes predictions?
12. How much do you trust the system's predictions?
13. How useful are the confidence scores for decision-making?
14. How helpful are the Grad-CAM visualizations for understanding predictions?
15. How likely are you to use this system in your actual research?

### Open-Ended Questions

16. What did you like most about the system?
17. What did you like least about the system?
18. What features or information are missing?
19. What would you change to make the system more useful for your work?
20. Any other comments or suggestions?

---

## Specific Metrics to Collect

### Quantitative Metrics
- **Task completion rate**: % of participants who successfully complete each task
- **Task completion time**: Average time for each task
- **Error rate**: Number of incorrect actions or misinterpretations
- **SUS score**: Overall usability score (target: >70)
- **Confidence threshold preference**: Distribution of thresholds participants would use

### Qualitative Observations
- **Common confusion points**: Where do users get stuck?
- **Mental model alignment**: Do users understand what the system does?
- **Trust factors**: What increases/decreases trust in predictions?
- **Feature usage**: Which features are used/ignored?
- **Terminology issues**: Any terms that are unclear?

---

## Key Questions to Answer

### Usability
1. Can users successfully upload data and run inference without assistance?
2. Do users understand the results table and how to navigate it?
3. Are filtering/sorting controls intuitive?
4. Is the workflow logical and efficient?

### Interpretability
5. Do users correctly understand what confidence scores represent?
6. Can users explain what Grad-CAM visualizations show?
7. Do users know how to act on different confidence levels?
8. Do users understand the system's limitations?

### Trust and Adoption
9. What confidence level do users consider "trustworthy"?
10. Would users rely on this system for research decisions?
11. Do visualizations increase or decrease trust?
12. What additional information do users need to trust predictions?

---

## Analysis Plan

### After Testing
1. **Compile notes** from all sessions
2. **Calculate SUS scores** and other quantitative metrics
3. **Identify patterns** in confusion points and suggestions
4. **Categorize issues** by severity:
   - Critical: Prevents task completion
   - Serious: Causes significant difficulty or errors
   - Minor: Small annoyances that don't prevent success
5. **Prioritize improvements** based on frequency and impact

### Reporting
- Executive summary with key findings
- Detailed findings for each task
- Recommendations ranked by priority
- Video clips of critical moments (with participant consent)

---

## Ethical Considerations

- Obtain informed consent before testing
- Ensure participant data privacy and anonymity
- Allow participants to stop at any time
- Compensate participants for their time (if applicable)
- Store recordings securely and delete after analysis (as per protocol)

---

## Appendix

### Test Data Package Contents
**File**: salmon_test_data.zip

**Contents**:
- 12 TIFF images: 6_2X.tiff, 14_2X.tiff, 42_2X.tiff, 136_2X.tiff, 256_2X.tiff, 478_2X.tiff, 521_2X.tiff, 522_2X.tiff, 597_2X.tiff, 624_2X.tiff, 664_2X.tiff, 741_2X.tiff
- 1 CSV file: selected_samples_info.csv
  - Columns: Scale_Id, Image_Id, Sort_Id, FL, Sex
  - 12 rows of metadata

**Expected Results**:
- All 12 images should match with CSV records
- Mix of Hatchery and Natural predictions
- Range of confidence scores for discussion

---

*End of Usability Test Protocol*
