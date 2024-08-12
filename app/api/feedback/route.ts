import OpenAI from 'openai';
import { NextResponse } from 'next/server';
import { streamText } from 'ai';
import { openai } from '@ai-sdk/openai';

type Prompt = {
  prompt: string;
}

export const runtime = 'edge';

export async function POST(req: Request): Promise<Response> {
  try {
    const { prompt: studentWriting, learningOutcomes, markingCriteria, promptType } = await req.json();

    // Calculate the length of the student's writing
    const studentWritingLength = studentWriting.length;

    // Define the maximum length for the feedback (twice the length of the student's writing)
    const maxFeedbackLength = studentWritingLength * 2;

    // Split the learning outcomes string into an array based on new lines
    const learningOutcomesArray = learningOutcomes.split('\n').filter((outcome: string) => outcome.trim() !== '');

    // Dynamically generate feedback sections based on learning outcomes
    const learningOutcomesFeedback = learningOutcomesArray.map((outcome: string, index: number) => {
      return `\n\n**Learning Outcome ${index + 1}:** ${outcome}\n\n- Provide feedback based on how well the student meets this outcome.\n\n---\n\n`;
    }).join('');

    // Construct the final prompt
    const generatePrompt = promptType === 'feedback' 
      ? `Using the provided learning outcomes, please evaluate the following student submission. Focus your feedback on each of the following learning outcomes:
      ${learningOutcomesFeedback}
      
      \n\nAfter addressing the learning outcomes, please provide comments on the following areas:
      \n\n- **Structure:** Evaluate the coherence, flow, and organization of ideas.
      \n\n- **Strengths:** Highlight the key strengths of the submission.
      \n\n- **Critical Thinking and Analysis:** Assess the student's ability to integrate and synthesize information, including the quality of argumentation and original insights.
      
      Ensure that the feedback is well-structured with each section separated by two line breaks (double newlines) and does not exceed twice the length of the student's writing, which is ${maxFeedbackLength} characters. Use Markdown formatting to clearly separate sections.`
      : 'Grade the student writing. Based on the expected learning outcomes and marking criteria, please provide a mark out of 100 for the following student writing.';

    console.log(studentWriting, learningOutcomes, markingCriteria, promptType);

    const result = await streamText({
      model: openai('gpt-3.5-turbo'),
      system: `
        Imagine yourself as a seasoned high school educator with extensive experience teaching across multiple academic disciplines, including mathematics, sciences, humanities, and arts. Your expertise is not only in delivering subject content but also in fostering critical thinking, creativity, and problem-solving skills among students. You are adept at using a variety of teaching methods, from traditional lectures to technology-integrated interactive sessions, to cater to diverse learning styles. Over the years, you have developed proficiency in creating engaging lesson plans, conducting rigorous assessments, and providing constructive feedback. Your role also involves mentoring students on academic and personal development, collaborating with colleagues to enhance interdisciplinary learning, and staying updated with the latest educational research and technology.`,
      messages: [
        {
          role: 'user',
          content: `
            You are a high school teacher responsible for teaching Geography.
            Here is the student writing: ${studentWriting}.
            Here are the expected learning outcomes: ${learningOutcomes}.
            Here is the marking criteria: ${markingCriteria}.
            You need to: ${generatePrompt}.
          `,
        },
      ],
      temperature: 0.4,
      presencePenalty: 0.3,
      frequencyPenalty: 0.5,
    });

    return result.toAIStreamResponse();
  } catch (error) {
    if (error instanceof OpenAI.APIError) {
      const { name, status, headers, message } = error;
      return NextResponse.json({ name, status, headers, message }, { status });
    } else {
      throw error;
    }
  }
}
