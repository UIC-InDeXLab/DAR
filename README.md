\begin{itemize}
    \item Graph on top of the $\eps$-net:
    \begin{itemize}
        \item[$\square$] Randomly assigning edges between nodes with some probabilities (Uniform and Gaussian).
        \item[$\square$] Theta-graph
        \item[$\square$] $k$NN graph (Nearest Neighbor Graph): NSW (\href{https://ieeexplore.ieee.org/abstract/document/8594636?casa_token=M8eVuxrYPi0AAAAA:gt-dvlmyyodNvhxEdsWRyUAKnj6hELYLLaJB3GtL1Pr3uLCVBIo5wYtqHTvn4AXouH040RNhKA}{hnsw})
    \end{itemize}
    \item To Report (hyper-parameters):
    \begin{itemize}
        \item Different sizes ($N$)
        \item Different dimensions: [2, 4, 8, 16, 32, 64 (opt.)]
        \item Different underlying points distribution: [Uniform, and Gaussian]
        \item Different sizes of $\eps$-net: [$N/2$, $N/4$, $N/16$, $N/64$, ...] (as small as $200$)
        \item Different number of points in the range: [different distances between parallel lines, or different sizes of output ($k$)]
        \item To Report:
        \begin{itemize}
            \item Query Time
            \item Space Usage
            \item Precision
        \end{itemize}
    \end{itemize}
\end{itemize}
\color{black}
