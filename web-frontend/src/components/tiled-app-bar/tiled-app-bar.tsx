import AppBar from "@mui/material/AppBar";
import Container from "@mui/material/Container";
import { Link } from "react-router-dom";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";

const TiledAppBar = () => {
  return (
    <AppBar position="static">
      <Container maxWidth="xl">
        <Toolbar disableGutters>
          <Box
            component={Link}
            to="/browse/"
            sx={{
              display: "flex",
              alignItems: "center",
              textDecoration: "none",
              color: "inherit",
              mr: 3,
            }}
          >
            <img
              src={`${import.meta.env.BASE_URL}tiled_logo.svg`}
              alt="Tiled logo"
              style={{ height: 28, marginRight: 8 }}
            />
            <Typography variant="h6" noWrap>
              TILED
            </Typography>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
};
export default TiledAppBar;
