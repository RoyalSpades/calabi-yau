from cy import plotly_quintic_param_explorer_surface

if __name__ == "__main__":
    plotly_quintic_param_explorer_surface(C_mag=1.0, phase_steps=12, branches="principal",
                                          nr=70, ntheta=140, opacity=0.5)