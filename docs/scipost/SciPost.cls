\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{SciPost} % SciPost Latex Template v1e (2017/10/31)


\LoadClass[11pt,a4paper]{article}


% Layout
\RequirePackage[top=12mm,bottom=12mm,left=30mm,right=30mm,head=12mm,includeheadfoot]{geometry}
\bigskipamount 6mm

% For table of contents: remove trailing dots
\RequirePackage{tocloft}
\renewcommand{\cftdot}{}
% Add References to TOC
\RequirePackage[nottoc,notlot,notlof]{tocbibind}


% Spacings between (sub)sections:
\RequirePackage{titlesec}
\titlespacing*{\section}{0pt}{1.8\baselineskip}{\baselineskip}


% Unicode characters
\RequirePackage[utf8]{inputenc}

% doi links in references
\RequirePackage{doi}

% Math formulas and symbols
%\RequirePackage{amsmath,amssymb} % Redundant (clashes with mathdesign)
\RequirePackage{amsmath}

% Hyperrefs
\RequirePackage{hyperref}

% Include line numbers in submissions
\RequirePackage{lineno}

% SciPost BiBTeX style
\bibliographystyle{SciPost_bibstyle}

% SciPost header and footer
\RequirePackage{fancyhdr}
\pagestyle{fancy}

\makeatletter
  \let\ps@plain\ps@fancy
\makeatother

\RequirePackage{xcolor}
\definecolor{scipostdeepblue}{HTML}{002B49}

\RequirePackage{graphicx}

\RequirePackage{cite}

%% Patch lineno when used with amsmath
\newcommand*\patchAmsMathEnvironmentForLineno[1]{%
\expandafter\let\csname old#1\expandafter\endcsname\csname #1\endcsname
\expandafter\let\csname oldend#1\expandafter\endcsname\csname end#1\endcsname
\renewenvironment{#1}%
{\linenomath\csname old#1\endcsname}%
{\csname oldend#1\endcsname\endlinenomath}}%
\newcommand*\patchBothAmsMathEnvironmentsForLineno[1]{%
\patchAmsMathEnvironmentForLineno{#1}%
\patchAmsMathEnvironmentForLineno{#1*}}%
\AtBeginDocument{%
\patchBothAmsMathEnvironmentsForLineno{equation}%
\patchBothAmsMathEnvironmentsForLineno{align}%
\patchBothAmsMathEnvironmentsForLineno{flalign}%
\patchBothAmsMathEnvironmentsForLineno{alignat}%
\patchBothAmsMathEnvironmentsForLineno{gather}%
\patchBothAmsMathEnvironmentsForLineno{multline}%
}
%% End patch lineno

\DeclareOption{Phys}{
\lhead{
  \colorbox{scipostdeepblue}{\strut \bf \color{white} ~SciPost Physics }
}
}

\DeclareOption{LectureNotes}{
  \lhead{
    \colorbox{scipostdeepblue}{\strut \bf \color{white} ~SciPost Physics Lecture Notes }
  }
}
\ProcessOptions\relax

\DeclareOption{Proceedings}{
  \lhead{
    \colorbox{scipostdeepblue}{\strut \bf \color{white} ~SciPost Physics Proceedings }
  }
}
\ProcessOptions\relax

\DeclareOption{submission}{
  \rhead{
    \colorbox{scipostdeepblue}{\strut \bf \color{white} ~Submission }
  }
 }
\ProcessOptions\relax
